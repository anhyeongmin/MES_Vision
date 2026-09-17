from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from filelock import FileLock, Timeout

from mes_vision.training.data import require, write_json
from .backend import GenerationConfig, parse_response
from .gpu import GpuCoordinator
from .queue import AnalysisQueue
from .snapshots import load_snapshot


def watch_parent(parent_pid):
    """Exit even if the supervisor was forcibly killed; never leave orphan GPU work."""
    def wait():
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            handle = kernel.OpenProcess(0x00100000, False, parent_pid)
            if handle:
                kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            os._exit(71)
        else:
            while os.getppid() == parent_pid: time.sleep(.2)
            os._exit(71)
    threading.Thread(target=wait, daemon=True).start()


def child_main(connection, project_root, backend, mock_behavior, parent_pid):
    watch_parent(parent_pid)
    model = None
    try:
        if backend == "qwen":
            from .region_backend import build_backend
            model = build_backend(project_root)
        else:
            require(backend == "mock", "unsupported backend")
            model = None
        connection.send({"type": "ready"})
        while True:
            job = connection.recv()
            if job is None: break
            if job.get("type") == "idle_memory":
                if backend == "qwen":
                    import torch
                    if job["release_cache"]:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info()
                else:
                    free, total = 16 * 1024**3, 24 * 1024**3
                connection.send({"type": "idle_memory", "free_bytes": free, "total_bytes": total})
                continue
            try:
                if backend == "mock":
                    require(job["kind"] == "synthetic", "mock cannot analyze real data")
                    if mock_behavior == "crash": os._exit(23)
                    if mock_behavior == "slow": time.sleep(30)
                    if mock_behavior == "delay": time.sleep(.4)
                    raw = "not JSON" if mock_behavior == "invalid" else '{"observation":"합성 시험용 추가 관찰입니다.","needs_review":true}'
                    response = {"raw_text": raw, "truncated": False, "deadline_expired": False,
                                "model": {"id": "explicit-synthetic-mock"}, "metrics": {}, "prompt": {}}
                else: response = model.generate(job)
                connection.send({"type": "result", "token": job["token"], "response": response})
            except Exception as exc:
                connection.send({"type": "error", "token": job["token"], "error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError): pass
    except Exception as exc:
        try: connection.send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        except (EOFError, BrokenPipeError): pass
    finally:
        try:
            if model is not None: model.close()
        finally: connection.close()


class ModelProcess:
    def __init__(self, root, backend, mock_behavior):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=child_main, args=(child, str(root), backend, mock_behavior, os.getpid()), daemon=True)
        self.process.start()
        child.close()

    def stop(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)
        require(not self.process.is_alive(), "model child could not be stopped; GPU ownership must not be released")
        self.connection.close()
        self.process.close()


class InterruptedWork(Exception):
    pass


class AnalysisWorker:
    def __init__(self, queue, project_root, *, backend="qwen", allow_synthetic=False, mock_behavior="ok",
                 load_timeout=120, idle_seconds=1, stop_event=None, keep_alive_seconds=30, min_free_mib=2048):
        require(backend in {"qwen", "mock"} and (backend != "mock" or allow_synthetic), "mock backend requires synthetic mode")
        require(mock_behavior in {"ok", "invalid", "slow", "crash", "delay"}, "invalid mock behavior")
        require(0 < load_timeout <= 300 and 0 <= idle_seconds <= 60, "invalid worker time settings")
        require(type(keep_alive_seconds) in {int, float} and 0 <= keep_alive_seconds <= 300,
                "keep_alive_seconds must be 0..300")
        require(type(min_free_mib) is int and 512 <= min_free_mib <= 32768, "min_free_mib must be 512..32768")
        self.queue, self.project_root = queue, Path(project_root)
        self.backend, self.allow_synthetic, self.mock_behavior = backend, allow_synthetic, mock_behavior
        self.load_timeout, self.idle_seconds = load_timeout, idle_seconds
        self.stop_event = stop_event or threading.Event()
        self.gpu = GpuCoordinator(queue.root / "gpu-coordination")
        self.owner = uuid4().hex
        self.child = None
        self.lock = None
        self.keep_alive_seconds, self.min_free_mib = keep_alive_seconds, min_free_mib
        self.warm_until = 0.
        self.memory_check_at = 0.
        self.idle_memory = None
        self.last_release_reason = None
        self.model_load_count = 0
        self.phase = "IDLE"
        self.phase_key = None
        self.phase_started = time.time()
        self.state_written_at = 0.
        self.preload_requested = threading.Event()
        self.unload_requested = threading.Event()
        self.preload_mutex = threading.Lock()
        self.preload_owners = set()
        self.preload_error = None
        self.model_is_ready = False
        self.waiting_idle_memory = False

    def request_preload(self, owner='default'):
        with self.preload_mutex:
            self.preload_owners.add(owner)
            self.preload_error = None
            self.unload_requested.clear()
            self.preload_requested.set()

    def release_preload(self, owner='default'):
        with self.preload_mutex:
            self.preload_owners.discard(owner)
            if not self.preload_owners:
                self.preload_requested.clear()
                self.unload_requested.set()

    def clear_preloads(self):
        with self.preload_mutex:
            self.preload_owners.clear()
            self.preload_requested.clear()

    def load_model(self, job=None):
        self.model_is_ready = False
        self.child = ModelProcess(self.project_root, self.backend, self.mock_behavior)
        self.state('MODEL_LOADING', job)
        require(self.wait_message(job,self.load_timeout)['type']=='ready','invalid model startup response')
        self.model_is_ready = True
        self.model_load_count += 1

    def state(self, phase, job=None):
        key = (phase, job["id"] if job else None, job.get("token") if job else None)
        now = time.monotonic()
        if key == self.phase_key and now-self.state_written_at < 1: return
        if key != self.phase_key: self.phase_started = time.time()
        self.phase, self.phase_key, self.state_written_at = phase, key, now
        self.queue.runtime({"worker_pid": os.getpid(),
            "model_pid": self.child.process.pid if self.child is not None else None,
            "phase": phase, "job_id": job["id"] if job is not None else None,
            "job_token": job.get("token") if job else None, "phase_started": self.phase_started,
            "backend": self.backend, "updated": time.time(),
            "keep_alive_seconds": self.keep_alive_seconds, "min_free_mib": self.min_free_mib,
            "warm_expires_at": time.time()+max(0., self.warm_until-time.monotonic()) if phase == "WARM_IDLE" and not self.preload_requested.is_set() else None,
            "idle_memory": self.idle_memory, "last_release_reason": self.last_release_reason,
            "model_ready": self.model_is_ready,
            "preload_requested": self.preload_requested.is_set(), "preload_error": self.preload_error,
            "model_load_count": self.model_load_count})

    def interruption(self, job=None):
        if self.stop_event.is_set(): return "WORKER_STOPPED"
        if not self.queue.enabled(): return "VLM_DISABLED"
        if job is None and self.unload_requested.is_set(): return 'PRELOAD_CANCELLED'
        if job is not None and self.queue.get(job["id"])["cancel_requested"]: return "USER_CANCELLED"
        if self.gpu.foreground_requested(): return "FOREGROUND_PRIORITY"
        if self.gpu.resident_blocks_vlm(): return "FOREGROUND_PRIORITY"
        return None

    def wait_message(self, job, timeout):
        deadline = time.monotonic()+timeout
        while True:
            cause = self.interruption(job)
            if (cause=='FOREGROUND_PRIORITY' and self.waiting_idle_memory
                    and self.preload_requested.is_set() and self.gpu.resident_allows_retention()):
                # Drain the already sent memory reply before yielding ownership;
                # otherwise it could be mistaken for the next inference reply.
                cause=None
            if cause: raise InterruptedWork(cause)
            if job is not None: self.state(self.phase, job)
            if time.monotonic() >= deadline: raise TimeoutError("VLM_TIME_LIMIT")
            if self.child.connection.poll(.05):
                message = self.child.connection.recv()
                if message["type"] == "error": raise RuntimeError(message["error"])
                return message
            if not self.child.process.is_alive(): raise RuntimeError("VLM_PROCESS_EXITED")

    def release_gpu(self, reason=None):
        # Process termination must finish before another inference owns the GPU.
        had_resources = self.child is not None or self.lock is not None
        if self.child is not None:
            self.child.stop()
            self.child = None
        if self.lock is not None:
            self.lock.release()
            self.lock = None
        self.warm_until = 0.
        self.model_is_ready = False
        self.memory_check_at = 0.
        self.idle_memory = None
        if reason is not None: self.last_release_reason = reason
        if had_resources: self.state("IDLE")

    def retain_idle_model(self):
        """Keep ownership while weights are resident; yield through the existing gate."""
        if self.child is None:
            self.release_gpu()
            self.state("IDLE")
            return
        if not self.child.process.is_alive():
            self.release_gpu("IDLE_PROCESS_EXITED")
            return
        if not self.preload_requested.is_set() and (not self.keep_alive_seconds or time.monotonic() >= self.warm_until):
            self.release_gpu("IDLE_EXPIRED")
            return
        if time.monotonic() < self.memory_check_at: return
        try:
            # Only the model child owns CUDA. Release unused generation cache once per idle period.
            self.child.connection.send({"type": "idle_memory", "release_cache": self.idle_memory is None})
            self.waiting_idle_memory=True
            try: memory = self.wait_message(None, 2)
            finally: self.waiting_idle_memory=False
            require(memory["type"] == "idle_memory", "invalid idle memory response")
            free, total = memory["free_bytes"], memory["total_bytes"]
            require(type(free) is int and type(total) is int and 0 <= free <= total and total > 0,
                    "invalid idle memory values")
            if free < self.min_free_mib * 1024**2:
                self.clear_preloads()
                self.preload_error = 'LOW_FREE_MEMORY'
                self.release_gpu("LOW_FREE_MEMORY")
                return
            if not self.preload_requested.is_set() and time.monotonic() >= self.warm_until:
                self.release_gpu("IDLE_EXPIRED")
                return
            self.idle_memory = {"free_mib": free / 1024**2, "total_mib": total / 1024**2}
            self.memory_check_at = time.monotonic()+1.
            self.state("WARM_IDLE")
        except InterruptedWork as exc:
            self.release_gpu(str(exc))
        except Exception:
            if self.preload_requested.is_set():
                self.clear_preloads(); self.preload_error='MEMORY_STATUS_UNAVAILABLE'
            self.release_gpu("MEMORY_STATUS_UNAVAILABLE")

    def run(self, *, max_jobs=None, idle_exit_seconds=None):
        completed = 0
        idle_since = time.monotonic()
        with FileLock(str(self.queue.root / "worker.lock"), timeout=0):
            self.queue.recover()
            self.state("IDLE")
            try:
                while not self.stop_event.is_set() and (max_jobs is None or completed < max_jobs):
                    if self.unload_requested.is_set():
                        self.release_gpu('PRELOAD_RELEASED')
                        self.unload_requested.clear()
                    cause = self.interruption()
                    if cause or self.gpu.recently_foreground(self.idle_seconds):
                        if cause == 'VLM_DISABLED': self.clear_preloads()
                        if (cause in (None,'FOREGROUND_PRIORITY') and self.model_is_ready
                                and self.preload_requested.is_set() and self.gpu.resident_allows_retention()):
                            # Idle weights can coexist with this approved resident. No CUDA
                            # command is sent until foreground inference releases the lock.
                            if self.lock is not None: self.lock.release(); self.lock=None
                        else: self.release_gpu(cause or "RECENT_FOREGROUND")
                        # Report the current reason, not an old requeued job's error text.
                        self.state("DISABLED" if cause == "VLM_DISABLED" else "WAITING_FOREGROUND")
                        if idle_exit_seconds is not None and time.monotonic()-idle_since >= idle_exit_seconds: break
                        self.stop_event.wait(.1)
                        continue
                    if self.child is not None and not self.child.process.is_alive():
                        self.release_gpu("IDLE_PROCESS_EXITED")
                    if self.child is not None and not self.preload_requested.is_set() and self.warm_until and time.monotonic() >= self.warm_until:
                        self.release_gpu("IDLE_EXPIRED")
                    if self.lock is None:
                        candidate = self.gpu.background_lock()
                        try: candidate.acquire(timeout=0)
                        except Timeout:
                            self.state("WAITING_GPU")
                            self.stop_event.wait(.1)
                            continue
                        self.lock = candidate
                        if self.interruption():
                            self.release_gpu()
                            continue
                    job = self.queue.claim(self.owner)
                    if job is None:
                        if self.preload_requested.is_set() and self.child is None:
                            try:
                                self.load_model()
                                self.warm_until=time.monotonic()+self.keep_alive_seconds
                            except InterruptedWork as exc:
                                self.release_gpu(str(exc))
                            except Exception as exc:
                                self.preload_error=f'{type(exc).__name__}: {exc}'
                                self.clear_preloads()
                                self.release_gpu('PRELOAD_FAILED')
                        self.retain_idle_model()
                        if idle_exit_seconds is not None and time.monotonic()-idle_since >= idle_exit_seconds: break
                        self.stop_event.wait(.1)
                        continue
                    idle_since = time.monotonic()
                    diagnostic = None
                    processing_started = time.perf_counter()
                    load_seconds = 0.
                    model_reused = self.child is not None
                    try:
                        require(job["kind"] == "real" or self.allow_synthetic, "synthetic job requires an explicitly synthetic worker")
                        require(self.backend != "mock" or job["kind"] == "synthetic", "mock cannot analyze real data")
                        load_snapshot(job["snapshot_path"], expected_digest=job["snapshot_digest"])
                        config = GenerationConfig(**job["payload"]["generation"])
                        if self.child is None:
                            load_started = time.perf_counter()
                            self.load_model(job)
                            load_seconds = time.perf_counter()-load_started
                        self.state("ANALYZING", job)
                        self.child.connection.send(job)
                        message = self.wait_message(job, config.timeout_seconds)
                        require(message["type"] == "result" and message["token"] == job["token"], "stale or mismatched model response")
                        diagnostic = message["response"]
                        diagnostic["metrics"].update(model_reused=model_reused, model_load_seconds=load_seconds,
                            worker_processing_seconds=time.perf_counter()-processing_started)
                        require(not diagnostic["deadline_expired"], "VLM_TIME_LIMIT")
                        require(not diagnostic["truncated"], "VLM_RESPONSE_TRUNCATED")
                        observation = parse_response(diagnostic["raw_text"])
                        manifest, _, _ = load_snapshot(job["snapshot_path"], expected_digest=job["snapshot_digest"])
                        limitations = []
                        if manifest["reference"] is None: limitations.append("NORMAL_REFERENCE_UNCONFIGURED")
                        if manifest["criteria"] is None: limitations.append("INSPECTION_CRITERIA_UNCONFIGURED")
                        if job["kind"] == "synthetic": limitations.append("SYNTHETIC_NOT_PRODUCT_VALIDATION")
                        result = {"schema_version": 1, "job_id": job["id"], "snapshot_digest": job["snapshot_digest"],
                                  "run_id": job["payload"]["run_id"], "frame_id": job["payload"]["frame_id"], "object_id": job["object_id"],
                                  "advisory_only": True, "base_decision": job["payload"]["base_decision"],
                                  "analysis": observation, "limitations": limitations, "model": diagnostic["model"],
                                  "metrics": diagnostic["metrics"], "prompt_version": diagnostic.get("prompt_version", job["payload"]["prompt_version"]),
                                  "generation": job["payload"]["generation"], "kind": job["kind"]}
                        attempts = self.queue.root / "attempts"
                        attempts.mkdir(exist_ok=True)
                        write_json(attempts / f"{job['id']}-{job['token']}.json", {"result": result, "diagnostic": diagnostic})
                        self.queue.finish(job["id"], job["token"], "COMPLETED", result=result)
                        completed += 1
                        self.warm_until = time.monotonic()+self.keep_alive_seconds if self.keep_alive_seconds else 0.
                        self.memory_check_at = 0.
                        self.idle_memory = None
                    except InterruptedWork as exc:
                        self.release_gpu(str(exc))
                        cause = str(exc)
                        state = "PENDING" if cause in {"FOREGROUND_PRIORITY", "WORKER_STOPPED"} else "CANCELLED"
                        self.queue.finish(job["id"], job["token"], state, error=cause)
                        if state == "CANCELLED": completed += 1
                    except Exception as exc:
                        self.release_gpu("ANALYSIS_ERROR")
                        self.queue.finish(job["id"], job["token"], "FAILED", error=f"{type(exc).__name__}: {exc}", result={"diagnostic": diagnostic} if diagnostic else None)
                        completed += 1
                    idle_since = time.monotonic()
            finally:
                self.release_gpu("WORKER_STOPPED")
                self.state("STOPPED")
        return completed

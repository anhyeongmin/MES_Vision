"""Read-only device discovery, isolated from native driver stalls."""
import multiprocessing as mp
import queue
import time


def collect_devices(output,driver='uvc'):
    from mes_vision.robot.magician import Magician
    from .camera import D405Camera
    from .uvc_camera import UVCCamera
    # Preserve serial-port results even if the camera driver subsequently stalls.
    for key, read in (("ports", Magician.ports), ("cameras", UVCCamera.devices if driver=='uvc' else D405Camera.devices)):
        try: output.put((key, read(), None))
        except Exception as exc: output.put((key, [], str(exc)))


def discover_devices(*, timeout=10., collector=None, driver='uvc'):
    context=mp.get_context("spawn"); output=context.Queue(4)
    process=context.Process(target=collector or collect_devices,args=(output,) if collector else (output,driver),daemon=True)
    result={"cameras":[],"ports":[],"errors":{}}
    pending={"cameras","ports"}; started=False
    try:
        process.start(); started=True; deadline=time.monotonic()+timeout
        while pending:
            remaining=deadline-time.monotonic()
            if remaining<=0: break
            try: key, rows, error=output.get(timeout=min(.1,remaining))
            except queue.Empty:
                if not process.is_alive(): break
                continue
            if key not in pending: continue
            pending.remove(key); result[key]=rows
            if error: result["errors"][key]=error
        for key in pending: result["errors"][key]="timeout_or_exit"
        return result
    finally:
        if started:
            process.join(timeout=.2)
            if process.is_alive(): process.terminate(); process.join(timeout=1.)
            if process.is_alive(): process.kill(); process.join(timeout=1.)
            if not process.is_alive(): process.close()
        output.cancel_join_thread(); output.close()

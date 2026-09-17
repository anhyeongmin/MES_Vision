"""Single-GPU local jobs. A resume is a new run linked to an immutable parent run."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import time

from .config import JobConfig
from .data import read_json, require, sha256, validate_dataset, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def verified_artifact(run: Path, entry: dict) -> Path:
    path = (run / entry["path"]).resolve()
    require(path.is_relative_to(run.resolve()), "artifact path escapes source run")
    require(sha256(path) == entry["sha256"], "checkpoint SHA256 mismatch")
    return path


def checkpoint_entry(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def moment_preview(states):
    for state in states:
        if "exp_avg" in state:
            return state["exp_avg"].detach().float().cpu().reshape(-1)[:8].tolist()
    return []


def preflight(config: JobConfig):
    data = validate_dataset(config.dataset_dir)
    require(data["metadata"]["role"] == config.role, "dataset role differs from training job")
    require((data["metadata"]["kind"] == "synthetic") == config.synthetic, "synthetic/real mode mismatch")
    require(sha256(Path(config.initial_weights)) == config.initial_sha256, "initial weights SHA256 mismatch")
    require(importlib.metadata.version("rfdetr") == "1.9.4", "RF-DETR version mismatch")
    return data


def resume_preflight(config: JobConfig, data: dict, parent: Path):
    import torch
    previous = read_json(parent / "run.json")
    require(previous["status"] in {"COMPLETED", "STOPPED", "INTERRUPTED", "FAILED"}, "parent run is still active or invalid")
    old_config = JobConfig(**previous["config"])
    require(old_config.compatibility() == config.compatibility(), "resume hyperparameters/role/seed differ; start a new fine-tune job instead")
    require(previous["dataset_fingerprint"] == data["fingerprint"], "dataset changed since parent training")
    path = verified_artifact(parent, previous["checkpoints"]["resume"])
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(saved.get("optimizer_states") and saved.get("lr_schedulers"), "full optimizer and scheduler state required; inference weights cannot resume training")
    completed_epochs = int(saved["epoch"]) + 1
    require(config.epochs > completed_epochs, "total epochs must exceed completed checkpoint epochs")
    summary = {"run": str(parent), "checkpoint_sha256": previous["checkpoints"]["resume"]["sha256"],
               "global_step": int(saved["global_step"]), "completed_epochs": completed_epochs,
               "optimizer_state_entries": sum(len(x["state"]) for x in saved["optimizer_states"]),
               "moment_preview": moment_preview(saved["optimizer_states"][0]["state"].values()),
               "scheduler_last_epoch": saved["lr_schedulers"][0]["last_epoch"]}
    del saved
    return previous, path, summary


def build_components(config, data, output, *, weights=None, resume=None, training=True):
    import torch
    from pytorch_lightning import seed_everything
    from rfdetr.config import RFDETRSmallConfig, TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
    require(torch.cuda.is_available(), "CUDA GPU required for this training configuration")
    if config.precision == "bf16-mixed":
        require(torch.cuda.is_bf16_supported(), "BF16 unsupported on this GPU")
    seed_everything(config.seed, workers=True)
    model_config = RFDETRSmallConfig(pretrain_weights=str(weights or config.initial_weights),
                                    num_classes=len(data["categories"]), device="cuda",
                                    amp=config.precision != "32-true")
    train_config = TrainConfig(dataset_dir=config.dataset_dir, output_dir=str(output),
                              dataset_file="roboflow", epochs=config.epochs, batch_size=config.batch_size,
                              grad_accum_steps=config.grad_accum_steps, lr=config.lr, lr_encoder=config.lr_encoder,
                              num_workers=config.num_workers, seed=config.seed, use_ema=False,
                              tensorboard=False, wandb=False, mlflow=False, run_test=False,
                              multi_scale=False, expanded_scales=False, scale_jitter=False,
                              progress_bar=None, save_dataset_grids=False, augmentation_backend="torchvision",
                              class_names=[item["name"] for item in data["categories"]],
                              resume=str(resume) if resume else None, checkpoint_interval=10, skip_best_epochs=0,
                              notes={"application": "MES Vision", "role": config.role,
                                     "synthetic": config.synthetic, "dataset_fingerprint": data["fingerprint"]})
    module = RFDETRModelModule(model_config, train_config)
    require(module.model.class_embed.weight.shape[0] == len(data["categories"]) + 1, "training head class count mismatch")
    module.strict_loading = True
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config, accelerator="gpu", devices=1,
                            precision=config.precision, include_training_callbacks=training,
                            enable_model_summary=False, num_sanity_val_steps=0, log_every_n_steps=1)
    return module, datamodule, trainer


def train(config: JobConfig, output: str | Path, *, resume_from=None, stop_after_epoch=None):
    import torch
    from pytorch_lightning import Callback
    output = Path(output).resolve()
    require(not output.is_relative_to(Path(config.dataset_dir).resolve()), "training output must be outside dataset")
    if stop_after_epoch is not None:
        require(type(stop_after_epoch) is int and stop_after_epoch > 0, "invalid stop epoch")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "VALIDATING", "started_at_utc": utc(),
                "config": config.snapshot(), "checkpoints": {}, "production_ready": False,
                "scope": "synthetic execution test" if config.synthetic else "training only; release acceptance pending"}
    write_json(output / "run.json", manifest)
    trainer = None
    started = time.perf_counter()
    class Audit(Callback):
        def __init__(self):
            self.steps = 0
            self.started = None
        def on_train_start(self, trainer, pl_module):
            state_entries = sum(len(x.state) for x in trainer.optimizers)
            self.started = {"global_step": trainer.global_step, "optimizer_state_entries": state_entries}
            self.started["moment_preview"] = moment_preview(trainer.optimizers[0].state.values())
            self.started["scheduler_last_epoch"] = trainer.lr_scheduler_configs[0].scheduler.state_dict()["last_epoch"]
            if manifest.get("resume"):
                require(trainer.global_step == manifest["resume"]["global_step"], "global step was not restored")
                require(state_entries == manifest["resume"]["optimizer_state_entries"], "optimizer state was not restored")
                require(self.started["moment_preview"] == manifest["resume"]["moment_preview"], "optimizer moments were not restored")
                require(self.started["scheduler_last_epoch"] == manifest["resume"]["scheduler_last_epoch"], "scheduler state was not restored")
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            gradient = pl_module.model.class_embed.weight.grad
            require(gradient is not None and torch.isfinite(gradient).all().item(), "missing/non-finite classifier gradients")
            self.steps += 1
        def on_train_epoch_end(self, trainer, pl_module):
            state = {"time_utc": utc(), "epoch": trainer.current_epoch + 1, "global_step": trainer.global_step,
                     "device": str(next(pl_module.parameters()).device)}
            write_json(output / "progress.json", state)
            if (output / "STOP").exists() or (stop_after_epoch is not None and trainer.current_epoch + 1 >= stop_after_epoch):
                trainer.should_stop = True
    audit = Audit()
    try:
        data = preflight(config)
        manifest["dataset_fingerprint"] = data["fingerprint"]
        manifest["dataset_metadata"] = data["metadata"]
        write_json(output / "dataset-snapshot.json", data)
        parent, resume_path, previous = None, None, None
        if resume_from:
            parent = Path(resume_from).resolve()
            previous, resume_path, resume_info = resume_preflight(config, data, parent)
            manifest["resume"] = resume_info
        manifest["status"] = "RUNNING"
        write_json(output / "run.json", manifest)
        module, datamodule, trainer = build_components(config, data, output / "engine", resume=resume_path)
        trainer.callbacks.append(audit)
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path, weights_only=True)
        require(not trainer.interrupted, "training interrupted; resume from last completed epoch")
        require(audit.steps > 0, "training performed no optimizer updates")
        manifest["audit"] = {"optimizer_updates_this_run": audit.steps, "at_train_start": audit.started,
                             "global_step": trainer.global_step, "gpu": torch.cuda.get_device_name(0),
                             "precision": config.precision}
        require(validate_dataset(config.dataset_dir)["fingerprint"] == data["fingerprint"], "dataset changed while training")
        current_best = output / "engine/checkpoint_best_regular.pth"
        require(current_best.is_file(), "validation produced no best checkpoint")
        best_callback = next(c for c in trainer.callbacks if type(c).__name__ == "BestModelCallback")
        require(best_callback.best_model_score is not None, "validation selection score missing")
        score = float(best_callback.best_model_score)
        chosen = current_best
        if previous and previous.get("selection", {}).get("validation_map", -1) > score:
            chosen = verified_artifact(parent, previous["checkpoints"]["inference"])
            score = previous["selection"]["validation_map"]
        selected = torch.load(chosen, map_location="cpu", weights_only=True)
        inference = output / "inference.pth"
        torch.save({"model": selected["model"], "model_name": "RFDETRSmall",
                    "model_config": module.model_config.model_dump(mode="json"),
                    "args": {"num_classes": len(data["categories"]), "class_names": [c["name"] for c in data["categories"]]},
                    "mes_vision": {"role": config.role, "synthetic": config.synthetic,
                                   "dataset_fingerprint": data["fingerprint"], "production_ready": False}}, inference)
        del selected
        manifest["checkpoints"]["inference"] = checkpoint_entry(output, inference)
        manifest["selection"] = {"metric": "val/mAP_50_95", "validation_map": score,
                                 "source": str(chosen), "uses_test_split": False}
        write_json(output / "model.json", {"role": config.role, "synthetic": config.synthetic,
                   "production_ready": False, "weights": manifest["checkpoints"]["inference"],
                   "categories": data["categories"], "product_id": data["metadata"]["product_id"],
                   "thresholds": None, "dataset_fingerprint": data["fingerprint"]})
        last = torch.load(output / "engine/last.ckpt", map_location="cpu", weights_only=True, mmap=True)
        manifest["completed_epochs"] = int(last["epoch"]) + 1
        del last
        manifest["status"] = "COMPLETED" if manifest["completed_epochs"] >= config.epochs else "STOPPED"
    except BaseException as exc:
        manifest["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) or (trainer is not None and trainer.interrupted) else "FAILED"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        last_path = output / "engine/last.ckpt"
        if last_path.is_file():
            manifest["checkpoints"]["resume"] = checkpoint_entry(output, last_path)
        manifest["finished_at_utc"] = utc()
        manifest["seconds"] = time.perf_counter() - started
        write_json(output / "run.json", manifest)
    return manifest


def evaluate(run: str | Path, output: str | Path, *, split="test"):
    require(split in {"valid", "test"}, "evaluation split must be valid or test")
    source, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "started_at_utc": utc(), "source_run": str(source), "split": split,
              "production_ready": False}
    write_json(output / "evaluation.json", report)
    try:
        manifest = read_json(source / "run.json")
        require(manifest["status"] in {"COMPLETED", "STOPPED"}, "source run did not finish successfully")
        config = JobConfig(**manifest["config"])
        weights = verified_artifact(source, manifest["checkpoints"]["inference"])
        data = preflight(config)
        require(data["fingerprint"] == manifest["dataset_fingerprint"], "dataset changed; evaluation requires a versioned new dataset workflow")
        module, datamodule, trainer = build_components(config, data, output / "engine", weights=weights, training=False)
        metrics = (trainer.test(module, datamodule=datamodule, verbose=False) if split == "test" else
                   trainer.validate(module, datamodule=datamodule, verbose=False))
        require(metrics and metrics[0], "evaluation produced no metrics")
        report.update(status="COMPLETED", metrics=metrics, weights_sha256=manifest["checkpoints"]["inference"]["sha256"],
                      dataset_fingerprint=data["fingerprint"], synthetic=config.synthetic,
                      scope="synthetic execution only; not product accuracy" if config.synthetic else "held-out detection metrics; not final inspection acceptance")
        require(validate_dataset(config.dataset_dir)["fingerprint"] == data["fingerprint"], "dataset changed during evaluation")
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at_utc"] = utc()
        write_json(output / "evaluation.json", report)
    return report

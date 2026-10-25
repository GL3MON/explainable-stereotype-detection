"""Mirror the live Kaggle trainer's configuration and metrics to W&B."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_directory", type=Path)
    args = parser.parse_args()
    root = args.job_directory.resolve()
    job = json.loads((root / "job.json").read_text())
    records = []
    while not records:
        records = (root / "live.log").read_text(errors="replace").splitlines()
        records = [json.loads(s.removeprefix("TRAINING_CONFIG ")) for s in records
                   if s.startswith("TRAINING_CONFIG ")]
        if not records:
            time.sleep(5)
    config = {**records[-1], "logging_source": "Kaggle live execution stream",
              "kaggle_url": job["url"], "kaggle_version": job["version"]}
    import wandb
    run_file = root / "wandb_run.json"
    existing = json.loads(run_file.read_text()) if run_file.exists() else {}
    run = wandb.init(
        project="stereodetect", name="qwen35-2b-sentence-labels-2t4",
        id=existing.get("id"), resume="allow" if existing else None,
        config=config, job_type="train", dir=str(root), mode="online",
        tags=["qwen3.5-2b", "sentence-only", "lora", "kaggle", "2xt4"],
        settings=wandb.Settings(x_disable_stats=True, disable_code=True, console="off"),
    )
    run_file.write_text(json.dumps({"id": run.id, "url": run.url,
                                  "entity": run.entity, "project": run.project}, indent=2) + "\n")
    print("W&B run:", run.url, flush=True)
    run.define_metric("optimizer_step")
    for pattern in ("train/*", "eval/*", "test/*"):
        run.define_metric(pattern, step_metric="optimizer_step")
    run.summary["status"] = "running"
    state_file = root / "wandb_bridge_state.json"
    seen = set(json.loads(state_file.read_text()).get("seen", [])) if state_file.exists() else set()
    while True:
        complete = False
        for line in (root / "live.log").read_text(errors="replace").splitlines():
            event, _, body = line.partition(" ")
            if event not in ("TRAIN", "VALIDATION", "COMPLETE", "BATCH_PROBE"):
                continue
            try:
                value = json.loads(body)
            except ValueError:
                continue
            key = event + ":" + str(value.get("step", "final") if isinstance(value, dict) else "batch")
            if key in seen:
                if event == "COMPLETE":
                    complete = True
                continue
            if event == "TRAIN":
                payload = {"optimizer_step": value["step"]}
                payload.update({"train/" + k: v for k, v in value.items()
                                if k != "step" and isinstance(v, (int, float))})
                run.log(payload)
                print("Logged step", value["step"], flush=True)
            elif event == "VALIDATION":
                payload = {"optimizer_step": value["step"], "eval/epoch": value["epoch"]}
                payload.update({"eval/" + k: v for k, v in value["validation"].items()
                                if isinstance(v, (int, float))})
                run.log(payload)
            elif event == "BATCH_PROBE":
                run.summary["batch_probe"] = value
            else:
                payload = {"optimizer_step": config["total_steps"]}
                payload.update({"test/" + k: v for k, v in value["metrics"].items()
                                if isinstance(v, (int, float))})
                run.log(payload)
                run.summary["best_epoch"] = value["best_epoch"]
                run.summary["best_validation_macro_f1"] = value["best_validation_macro_f1"]
                run.summary["test/classification_report"] = value["metrics"]["classification_report"]
                run.summary["test/confusion_matrix"] = value["metrics"]["confusion_matrix"]
                complete = True
            seen.add(key)
            state_file.write_text(json.dumps({"seen": sorted(seen)}, indent=2) + "\n")
        if complete:
            run.summary["status"] = "completed"
            run.finish()
            print("W&B metrics complete", flush=True)
            return
        monitor = root / "monitor_status.json"
        if monitor.exists() and json.loads(monitor.read_text())["state"] in ("ERROR", "CANCEL_ACKNOWLEDGED"):
            run.summary["status"] = "failed"
            run.finish(exit_code=1)
            return
        time.sleep(5)


if __name__ == "__main__":
    main()

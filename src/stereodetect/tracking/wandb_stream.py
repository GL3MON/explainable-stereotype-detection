"""Log an already-running training job to W&B without restarting it."""

import argparse
import ast
import json
import math
import re
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/qwen35_2b_stereodetect")
    parser.add_argument("--log", default="outputs/training.log")
    parser.add_argument("--project", default="stereodetect")
    parser.add_argument("--name", default="qwen35-2b-no-thinking")
    args = parser.parse_args(argv)
    import wandb

    output = Path(args.output_dir)
    config = json.loads((output / "run_config.json").read_text())
    config.update(
        {
            "logging_source": "live training log",
            "learning_rate": 2e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "seed": 42,
            "precision": "bf16",
            "gradient_checkpointing": True,
            "effective_batch_size": config["batch_size"] * config["gradient_accumulation"],
        }
    )
    steps_per_epoch = math.ceil(config["train_rows"] / config["effective_batch_size"])
    run = wandb.init(project=args.project, name=args.name, config=config, job_type="train")
    (output / "wandb_run.json").write_text(
        json.dumps({"id": run.id, "url": run.url, "project": args.project}, indent=2)
    )
    print(f"W&B run: {run.url}", flush=True)
    run.define_metric("optimizer_step")
    for pattern in ("train/*", "eval/*", "test/*"):
        run.define_metric(pattern, step_metric="optimizer_step")
    seen = 0
    while True:
        log = Path(args.log).read_text(errors="replace")
        entries = re.findall(r"\{'(?:loss|eval_loss)'[^\r\n]*?\}", log)
        for entry in entries[seen:]:
            values = ast.literal_eval(entry)
            step = round(float(values.get("epoch", 0)) * steps_per_epoch)
            payload = {"optimizer_step": step}
            for key, value in values.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if key.startswith("eval_"):
                    payload["eval/" + key[5:]] = value
                else:
                    payload["train/" + key] = value
            run.log(payload)
            print(f"Logged optimizer step {step}: {payload}", flush=True)
        seen = len(entries)
        evaluation = output / "evaluation.json"
        if evaluation.exists():
            report = json.loads(evaluation.read_text())
            payload = {"optimizer_step": round(config["epochs"] * steps_per_epoch)}
            payload.update(
                {
                    "test/" + key.removeprefix("test_"): value
                    for key, value in report["test_metrics"].items()
                }
            )
            run.log(payload)
            run.summary["best_validation_macro_f1"] = report["best_validation_macro_f1"]
            run.summary["best_checkpoint"] = report["best_checkpoint"]
            for label, values in report["classification_report"].items():
                if isinstance(values, dict):
                    for metric, value in values.items():
                        run.summary[f"test/{label}/{metric}"] = value
            run.summary["confusion_matrix"] = report["confusion_matrix"]
            run.summary["status"] = "completed"
            run.finish()
            print("W&B logging completed.", flush=True)
            break
        if "Traceback (most recent call last)" in log:
            run.summary["status"] = "failed"
            run.finish(exit_code=1)
            raise RuntimeError("Training failed; see " + args.log)
        time.sleep(10)


if __name__ == "__main__":
    main()

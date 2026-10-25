"""Evaluate generated category/target/label JSON; report metrics for labels only."""

import argparse
import csv
import json
from pathlib import Path

from stereodetect.models.qwen.json_task import (
    MODEL,
    SYSTEM,
    detailed_report,
    load_rows,
    messages,
    parse_label,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/qwen35_2b_stereodetect_json/adapter")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", default="outputs/eval/json_test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stereodetect")
    args = parser.parse_args(argv)
    from stereodetect.runtime import configure_runtime

    configure_runtime()
    import torch
    from peft import PeftConfig, PeftModel
    from sklearn.metrics import classification_report
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, set_seed

    if min(args.batch_size, args.max_new_tokens) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, max new tokens, and limit must be positive")
    if not (Path(args.adapter) / "adapter_config.json").is_file():
        parser.error(
            "No JSON-task adapter found; train the updated model first or specify --adapter PATH"
        )
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    rows = load_rows(Path(args.data_dir) / f"{args.split}.csv")
    total = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "completions.jsonl"
    if path.exists():
        parser.error(f"{path} exists; use a different --output-dir to preserve completions")
    model_id = (
        PeftConfig.from_pretrained(args.adapter, local_files_only=True).base_model_name_or_path
        or MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).to("cuda").eval()
    set_seed(42)
    records = []
    with path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for row in batch
            ]
            inputs = tokenizer(
                prompts, add_special_tokens=False, padding=True, return_tensors="pt"
            ).to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            texts = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for row, text in zip(batch, texts):
                label = parse_label(text)
                record = {
                    **row,
                    "completion": text,
                    "predicted_label": label,
                    "correct": label == int(row["labels"]),
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Evaluated {len(records)}/{len(rows)}", flush=True)
    truth = [int(row["labels"]) for row in rows]
    predicted = [row["predicted_label"] for row in records]
    report = {
        "model": model_id,
        "adapter": args.adapter,
        "split": args.split,
        "dataset_rows": total,
        "num_examples": len(rows),
        "partial_evaluation": len(rows) < total,
        "scored_field": "label",
        "enable_thinking": False,
        "system_prompt": SYSTEM,
        **detailed_report(truth, predicted),
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text_report = "Classification Report (labels only):\n" + classification_report(
        truth, predicted, labels=list(range(5)), digits=4, zero_division=0
    )
    text_report += f"\nAccuracy (all examples): {report['accuracy']:.4f}\nInvalid label predictions: {report['invalid_predictions']}\n"
    (output / "classification_report.txt").write_text(text_report)
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"json-{args.split}-label-eval",
            job_type="eval",
            config=vars(args),
        )
        run.log(
            {
                f"{args.split}/{key}": report[key]
                for key in (
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                )
            }
        )
        run.summary["classification_report"] = report["classification_report"]
        run.summary["confusion_matrix"] = report["confusion_matrix"]
        run.summary["invalid_predictions"] = report["invalid_predictions"]
        (output / "wandb_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n"
        )
        run.finish()
    print(text_report, flush=True)
    print(f"Full JSON completions: {path}", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate Qwen3.5-2B label predictions without generating reasoning."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

from stereodetect.data import load_label_rows as load_rows
from stereodetect.labels import LABELS
from stereodetect.metrics import calculate_metrics
from stereodetect.models.qwen.labels import LABEL_NAMES, MODEL, messages, sentence_only_for_adapter


def evaluate(model, tokenizer, rows, batch_size, max_length, device, sentence_only=False):
    import torch

    label_ids = [tokenizer.encode(str(label), add_special_tokens=False) for label in LABELS]
    if any(len(tokens) != 1 for tokens in label_ids):
        raise ValueError("Expected labels 0-4 to each be a single token")
    label_ids = [tokens[0] for tokens in label_ids]
    predictions, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    messages(row, sentence_only=sentence_only),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for row in batch
            ]
            inputs = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if inputs["input_ids"].shape[1] > max_length:
                raise ValueError(
                    f"Batch starting at example {start + 1} exceeds {max_length} tokens; increase --max-length"
                )
            # Left padding places every final prompt token at the last position.
            # Gold labels are used only for scoring after prediction.
            inputs = inputs.to(device)
            logits = (
                model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, label_ids].float()
            )
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite label scores in batch starting at example {start + 1}"
                )
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            probabilities.extend(logits.softmax(dim=-1).cpu().tolist())
            print(
                f"Evaluated {min(start + batch_size, len(rows))}/{len(rows)} examples", flush=True
            )
    return predictions, probabilities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--adapter",
        default="outputs/qwen35_2b_stereodetect/adapter",
        help="Saved LoRA adapter or checkpoint directory",
    )
    source.add_argument(
        "--base-only", action="store_true", help="Evaluate the cached base model without an adapter"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the base model ID/path; defaults to the adapter configuration",
    )
    parser.add_argument("--sentence-only", action="store_true", help="Use only Sentence as input")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--csv", default=None, help="Evaluate a custom CSV instead of --data-dir/--split"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples for a quick check",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--wandb", action="store_true", help="Log metrics and a confusion matrix to W&B"
    )
    parser.add_argument("--wandb-project", default="stereodetect")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args(argv)
    from stereodetect.runtime import configure_runtime

    configure_runtime()
    import torch
    from peft import PeftConfig, PeftModel
    from sklearn.metrics import classification_report
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if args.batch_size < 1 or args.max_length < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size, --max-length, and --limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu if needed")
    data_path = Path(args.csv) if args.csv else Path(args.data_dir) / f"{args.split}.csv"
    split = data_path.stem if args.csv else args.split
    try:
        rows, fields = load_rows(data_path)
        adapter = None if args.base_only else Path(args.adapter)
        if adapter is not None:
            if not (adapter / "adapter_config.json").is_file():
                parser.error(
                    f"No saved adapter at {adapter}. Provide --adapter PATH after training, or --base-only for a baseline."
                )
            if not any(
                (adapter / name).is_file()
                for name in ("adapter_model.safetensors", "adapter_model.bin")
            ):
                parser.error(f"Adapter weights are missing from {adapter}")
            config = PeftConfig.from_pretrained(str(adapter), local_files_only=True)
            model_id = args.model or config.base_model_name_or_path
        else:
            model_id = args.model or MODEL
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sentence_only = args.sentence_only or sentence_only_for_adapter(adapter)
    total_rows = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    source_name = "base" if adapter is None else adapter.name
    output = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/eval") / f"{source_name}_{split}"
    )
    if output.resolve() == data_path.resolve().parent:
        parser.error("--output-dir must differ from the dataset directory")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_source = (
        str(adapter)
        if adapter is not None and (adapter / "tokenizer_config.json").is_file()
        else model_id
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        local_files_only=True,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    model = model.to(args.device)
    started = time.perf_counter()
    predicted, probabilities = evaluate(
        model, tokenizer, rows, args.batch_size, args.max_length, args.device,
        sentence_only=sentence_only,
    )
    elapsed = time.perf_counter() - started
    truth = [int(row["labels"]) for row in rows]
    metrics = calculate_metrics(truth, predicted)
    metrics["label_log_loss"] = sum(
        -math.log(max(probs[label], 1e-15)) for label, probs in zip(truth, probabilities)
    ) / len(rows)
    by_category = {}
    for category in sorted({row["Category"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["Category"] == category]
        category_metrics = calculate_metrics(
            [truth[i] for i in indices], [predicted[i] for i in indices]
        )
        by_category[category] = {
            "num_examples": len(indices),
            **{key: category_metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
        }
    report = {
        "model": model_id,
        "adapter": str(adapter) if adapter else None,
        "dataset": str(data_path),
        "split": split,
        "dataset_rows": total_rows,
        "num_examples": len(rows),
        "partial_evaluation": len(rows) < total_rows,
        "enable_thinking": False,
        "sentence_only": sentence_only,
        "input_columns": ["Sentence"] if sentence_only else ["Category", "Target", "Sentence"],
        "prediction_method": "argmax over label-token logits 0-4",
        "probability_definition": "softmax normalized over the five label tokens; not calibrated confidence",
        "label_names": LABEL_NAMES,
        "confusion_matrix_axes": "rows=true labels, columns=predicted labels; order=0,1,2,3,4",
        "category_macro_f1_definition": "unweighted mean over all five labels, including absent classes as zero",
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "metrics": metrics,
        "by_category": by_category,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        extra = ["prediction", "prediction_name", "correct"] + [f"probability_{i}" for i in LABELS]
        if set(fields) & set(extra):
            raise ValueError("CSV columns conflict with prediction output columns")
        writer = csv.DictWriter(handle, fieldnames=fields + extra)
        writer.writeheader()
        for row, label, probs in zip(rows, predicted, probabilities):
            writer.writerow(
                {
                    **row,
                    "prediction": label,
                    "prediction_name": LABEL_NAMES[label],
                    "correct": int(label == int(row["labels"])),
                    **{f"probability_{i}": probs[i] for i in LABELS},
                }
            )
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{source_name}-{split}-eval",
            job_type="eval",
            config={
                **vars(args),
                "model": model_id,
                "enable_thinking": False,
                "dataset_rows": total_rows,
                "num_examples": len(rows),
                "partial_evaluation": len(rows) < total_rows,
            },
        )
        run.log(
            {
                f"{split}/{key}": metrics[key]
                for key in ("accuracy", "macro_f1", "weighted_f1", "label_log_loss")
            }
        )
        run.log(
            {
                f"{split}/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=truth, preds=predicted, class_names=LABEL_NAMES
                )
            }
        )
        for category, values in by_category.items():
            run.summary[f"{split}/category/{category}"] = values
        run.summary[f"{split}/classification_report"] = metrics["classification_report"]
        (output / "wandb_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n"
        )
        print(f"W&B: {run.url}", flush=True)
        run.finish()
    print(f"\nEvaluated {len(rows)}/{total_rows} examples from {data_path}")
    for key in ("accuracy", "macro_f1", "weighted_f1", "label_log_loss"):
        print(f"{key}: {metrics[key]:.4f}")
    print("\nClassification Report:")
    print(classification_report(truth, predicted, labels=LABELS, zero_division=0, digits=4))
    print(f"Report: {output / 'evaluation.json'}\nPredictions: {output / 'predictions.csv'}")


if __name__ == "__main__":
    main()

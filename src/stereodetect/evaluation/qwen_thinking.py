"""Generate thinking completions and evaluate final labels on StereoDetect."""

import argparse
import csv
import json
import re
import time
from pathlib import Path

from stereodetect.data import load_label_rows as load_rows
from stereodetect.metrics import calculate_metrics
from stereodetect.models.qwen.labels import MODEL, SYSTEM, messages

THINKING_SYSTEM = SYSTEM.replace(
    "Respond with exactly one digit from 0 to 4. Do not include reasoning or explanation.",
    "Think through the classification inside the thinking block. After </think>, respond with exactly one digit from 0 to 4.",
)


def parse_completion(generated_text):
    if "</think>" not in generated_text:
        return generated_text, "", -1
    reasoning, final = generated_text.rsplit("</think>", 1)
    for special in ("<|im_end|>", "<|endoftext|>"):
        final = final.split(special, 1)[0]
    final = final.strip()
    return reasoning.strip(), final, int(final) if re.fullmatch(r"[0-4]", final) else -1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/qwen35_2b_stereodetect/adapter")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/eval/adapter_test_thinking")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
        parser.error("Batch size, token budget, and limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        parser.error(f"No adapter at {adapter}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completions_path = output / "completions.jsonl"
    if completions_path.exists():
        parser.error(
            f"{completions_path} already exists; choose a new --output-dir to preserve previous completions"
        )
    data_path = Path(args.data_dir) / f"{args.split}.csv"
    rows, fields = load_rows(data_path)
    total_rows = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    config = PeftConfig.from_pretrained(str(adapter), local_files_only=True)
    model_id = config.base_model_name_or_path or MODEL
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter), local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True).to("cuda").eval()
    set_seed(42)
    records = []
    started = time.perf_counter()
    run_config = {
        **vars(args),
        "model": model_id,
        "enable_thinking": True,
        "system_prompt": THINKING_SYSTEM,
        "decoding": "greedy",
        "seed": 42,
        "dataset_rows": total_rows,
        "num_examples": len(rows),
        "invalid_prediction": -1,
        "label_extraction": "single digit after closing </think>; otherwise invalid",
    }
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    with completions_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = []
            for row in batch:
                msg = messages(row)
                msg[0]["content"] = THINKING_SYSTEM
                prompts.append(
                    tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                )
            inputs = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            for offset, (row, tokens) in enumerate(
                zip(batch, generated[:, inputs["input_ids"].shape[1] :])
            ):
                ids = tokens.cpu().tolist()
                ended = tokenizer.eos_token_id in ids
                if ended:
                    ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(ids, skip_special_tokens=False)
                reasoning, final, prediction = parse_completion(text)
                record = {
                    "index": start + offset,
                    **row,
                    "completion": "<think>\n" + text,
                    "generated_text": text,
                    "reasoning": reasoning,
                    "final_answer": final,
                    "prediction": prediction,
                    "correct": prediction == int(row["labels"]),
                    "invalid": prediction == -1,
                    "thinking_closed": "</think>" in text,
                    "truncated": not ended and len(ids) >= args.max_new_tokens,
                    "generated_tokens": len(ids),
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.perf_counter() - started
            print(
                f"Evaluated {len(records)}/{len(rows)}; invalid={sum(r['invalid'] for r in records)}; "
                f"truncated={sum(r['truncated'] for r in records)}; elapsed={elapsed:.1f}s",
                flush=True,
            )
    truth = [int(row["labels"]) for row in rows]
    predictions = [record["prediction"] for record in records]
    metrics = calculate_metrics(truth, predictions)
    # Include invalid predictions as a separate column; every example counts toward accuracy/recall.
    matrix = [[0] * 6 for _ in range(5)]
    for actual, predicted in zip(truth, predictions):
        matrix[actual][predicted if predicted != -1 else 5] += 1
    metrics["confusion_matrix"] = matrix
    report = {
        **run_config,
        "partial_evaluation": len(rows) < total_rows,
        "metrics": metrics,
        "invalid_completions": sum(r["invalid"] for r in records),
        "truncated_completions": sum(r["truncated"] for r in records),
        "thinking_closed_completions": sum(r["thinking_closed"] for r in records),
        "average_generated_tokens": sum(r["generated_tokens"] for r in records) / len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "confusion_matrix_axes": "rows=gold 0-4; columns=prediction 0-4, invalid",
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text_report = (
        "Classification Report:\n"
        + classification_report(
            truth, predictions, labels=list(range(5)), digits=4, zero_division=0
        )
        + f"\nAccuracy (all examples): {metrics['accuracy']:.4f}\n"
        f"Invalid completions: {report['invalid_completions']}\n"
        f"Truncated completions: {report['truncated_completions']}\n"
    )
    (output / "classification_report.txt").write_text(text_report)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        extra = [
            "prediction",
            "correct",
            "invalid",
            "thinking_closed",
            "truncated",
            "generated_tokens",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fields + extra} for r in records)
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"adapter-{args.split}-thinking-eval",
            job_type="eval",
            config=run_config,
        )
        run.log(
            {f"{args.split}/{key}": metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1")}
        )
        for key in (
            "invalid_completions",
            "truncated_completions",
            "average_generated_tokens",
            "thinking_closed_completions",
        ):
            run.summary[key] = report[key]
        run.summary["classification_report"] = metrics["classification_report"]
        run.summary["confusion_matrix"] = matrix
        (output / "wandb_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n"
        )
        print(f"W&B: {run.url}", flush=True)
        run.finish()
    print(text_report, flush=True)
    print(f"Full completions: {completions_path}", flush=True)


if __name__ == "__main__":
    main()

"""Train Qwen3.5-2B to generate category/target/label JSON; score labels only."""

import argparse
import csv
import json
import os
from pathlib import Path

from stereodetect.models.qwen.data import Collator
from stereodetect.models.qwen.json_task import (
    CATEGORIES,
    MODEL,
    SYSTEM,
    detailed_report,
    label_metrics,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/qwen35_2b_stereodetect_json")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values run a training check without evaluation",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-project", default="stereodetect")
    parser.add_argument("--wandb-run-name", default="qwen35-2b-json-label-metrics")
    args = parser.parse_args(argv)
    from stereodetect.runtime import configure_runtime

    configure_runtime()
    import torch
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import classification_report
    from transformers import (
        AutoTokenizer,
        Qwen3_5ForConditionalGeneration,
        TrainingArguments,
        set_seed,
    )

    from stereodetect.models.qwen.data import JsonExamples as Examples
    from stereodetect.models.qwen.trainer import JsonTrainer

    if (
        min(
            args.batch_size,
            args.eval_batch_size,
            args.gradient_accumulation,
            args.max_length,
            args.max_new_tokens,
        )
        < 1
        or args.epochs <= 0
        or args.max_steps == 0
    ):
        parser.error(
            "Batch sizes, token budgets, accumulation, and epochs must be positive; max steps must be positive or -1"
        )
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    set_seed(42)
    output = Path(args.output_dir)
    if output.resolve() == Path("outputs/qwen35_2b_stereodetect").resolve():
        parser.error(
            "Choose a different output directory to preserve the previous label-only model"
        )
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, val, test = [
        Examples(Path(args.data_dir) / f"{split}.csv", tokenizer, args.max_length)
        for split in ("train", "val", "test")
    ]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=r"model.language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)",
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no" if args.max_steps > 0 else "epoch",
        save_strategy="no" if args.max_steps > 0 else "epoch",
        save_total_limit=2,
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none" if args.max_steps > 0 else "wandb",
        run_name=args.wandb_run_name,
        optim="adamw_torch",
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
    )
    trainer = JsonTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=Collator(tokenizer.pad_token_id),
        max_new_tokens=args.max_new_tokens,
        compute_metrics=lambda p: label_metrics(
            p.label_ids.reshape(-1).tolist(), p.predictions.reshape(-1).tolist()
        ),
    )
    config = {
        **vars(args),
        "model": MODEL,
        "enable_thinking": False,
        "input_columns": ["Sentence"],
        "output_keys": ["category", "target", "label"],
        "system_prompt": SYSTEM,
        "category_normalization": "strip, casefold, collapse whitespace",
        "categories": CATEGORIES,
        "target_normalization": "none",
        "scored_field": "label",
        "invalid_label": -1,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=args.resume)
    if args.max_steps > 0:
        print("Training check passed:", result.metrics, flush=True)
        return
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    predictions = trainer.predict(test, metric_key_prefix="test")
    trainer.save_metrics("test", predictions.metrics)
    trainer.log(predictions.metrics)
    truth = predictions.label_ids.reshape(-1).tolist()
    guessed = predictions.predictions.reshape(-1).tolist()
    report = {
        "scored_field": "label",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "test_metrics": predictions.metrics,
        **detailed_report(truth, guessed),
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    completions = [{**row, **record} for row, record in zip(test.rows, trainer.completions)]
    (output / "test_completions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completions)
    )
    text_report = "Classification Report (labels only):\n" + classification_report(
        truth, guessed, labels=list(range(5)), zero_division=0, digits=4
    )
    text_report += f"\nAccuracy (all examples): {report['accuracy']:.4f}\nInvalid label predictions: {report['invalid_predictions']}\n"
    (output / "classification_report.txt").write_text(text_report)
    with (output / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(test.rows[0]) + ["completion", "predicted_label", "correct"]
        )
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "completion": record["completion"],
                "predicted_label": record["predicted_label"],
                "correct": record["correct"],
            }
            for row, record in zip(test.rows, trainer.completions)
        )
    print(text_report, flush=True)


if __name__ == "__main__":
    main()

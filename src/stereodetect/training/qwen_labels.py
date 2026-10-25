"""Offline, non-thinking LoRA fine-tuning for StereoDetect's five labels."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from stereodetect.models.qwen.labels import LABEL_NAMES, MODEL, SENTENCE_SYSTEM, SYSTEM


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--sentence-only", action="store_true", help="Use only Sentence to predict labels 0-4"
    )
    parser.add_argument("--report-to", choices=["wandb", "none"], default="wandb")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-project", default="stereodetect")
    parser.add_argument("--wandb-run-name", default="qwen35-2b-no-thinking")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            "outputs/qwen35_2b_stereodetect_sentence_labels"
            if args.sentence_only else "outputs/qwen35_2b_stereodetect"
        )
    if min(args.batch_size, args.gradient_accumulation, args.max_length) < 1 or args.epochs <= 0:
        parser.error("Batch size, accumulation, maximum length, and epochs must be positive")
    from stereodetect.runtime import configure_runtime

    configure_runtime()
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import (
        AutoTokenizer,
        Qwen3_5ForConditionalGeneration,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from stereodetect.models.qwen.data import Collator
    from stereodetect.models.qwen.data import LabelExamples as Examples

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training run.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(5)]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Expected single-token numeric labels")
    label_ids = [ids[0] for ids in label_ids]
    train = Examples(
        Path(args.data_dir) / "train.csv", tokenizer, args.max_length,
        sentence_only=args.sentence_only,
    )
    val = Examples(
        Path(args.data_dir) / "val.csv", tokenizer, args.max_length,
        sentence_only=args.sentence_only,
    )
    test = Examples(
        Path(args.data_dir) / "test.csv", tokenizer, args.max_length,
        sentence_only=args.sentence_only,
    )
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

    def reduce_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        first = (labels != -100).int().argmax(dim=1)
        return logits[torch.arange(logits.shape[0], device=logits.device), first - 1][
            :, label_ids
        ].float()

    def metrics(pred):
        first = (pred.label_ids != -100).argmax(axis=1)
        truth_tokens = pred.label_ids[np.arange(len(first)), first]
        truth = np.array([label_ids.index(int(token)) for token in truth_tokens])
        guessed = pred.predictions.argmax(axis=1)
        return {
            "accuracy": accuracy_score(truth, guessed),
            "macro_f1": f1_score(
                truth, guessed, labels=list(range(5)), average="macro", zero_division=0
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        warmup_steps=0.05,
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
        report_to="none" if args.max_steps > 0 else args.report_to,
        run_name=args.wandb_run_name,
        optim="adamw_torch",
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=Collator(tokenizer.pad_token_id),
        compute_metrics=metrics,
        preprocess_logits_for_metrics=reduce_logits,
    )
    (output / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "model": MODEL,
                "enable_thinking": False,
                "label_names": LABEL_NAMES,
                "system_prompt": SENTENCE_SYSTEM if args.sentence_only else SYSTEM,
                "input_columns": ["Sentence"] if args.sentence_only else ["Category", "Target", "Sentence"],
                "output_format": "one digit 0-4",
                "learning_rate": training_args.learning_rate,
                "effective_batch_size": args.batch_size * args.gradient_accumulation,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "seed": 42,
                "data_sha256": {
                    split: hashlib.sha256((Path(args.data_dir) / f"{split}.csv").read_bytes()).hexdigest()
                    for split in ("train", "val", "test")
                },
                "evaluation": "Argmax of the five label-token logits, with thinking disabled",
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
            },
            indent=2,
        )
    )
    result = trainer.train(resume_from_checkpoint=args.resume)
    if args.max_steps > 0:
        print("GPU training check passed:", result.metrics, flush=True)
        return
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    (output / "adapter" / "run_config.json").write_text((output / "run_config.json").read_text())
    predictions = trainer.predict(test, metric_key_prefix="test")
    trainer.save_metrics("test", predictions.metrics)
    trainer.log(predictions.metrics)
    guessed = predictions.predictions.argmax(axis=1)
    truth = np.array([int(row["labels"]) for row in test.rows])
    report = {
        "enable_thinking": False,
        "sentence_only": args.sentence_only,
        "input_columns": ["Sentence"] if args.sentence_only else ["Category", "Target", "Sentence"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "test_metrics": predictions.metrics,
        "classification_report": classification_report(
            truth,
            guessed,
            labels=list(range(5)),
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(truth, guessed, labels=list(range(5))).tolist(),
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    (output / "classification_report.txt").write_text(classification_report(
        truth, guessed, labels=list(range(5)), target_names=LABEL_NAMES,
        digits=4, zero_division=0,
    ))
    with open(output / "test_predictions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test.rows[0]) + ["prediction"])
        writer.writeheader()
        writer.writerows(
            {**row, "prediction": int(prediction)} for row, prediction in zip(test.rows, guessed)
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

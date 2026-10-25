"""Fine-tune ModernBERT-base on StereoDetect's five labels using sentences only."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from stereodetect.labels import LABEL_NAMES
from stereodetect.metrics import detailed_metrics, metrics
from stereodetect.runtime import package_versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir", default="outputs/modernbert_stereodetect_lr5e5_cosine_ep3_seed3407"
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-strategy", choices=["epoch", "steps"], default="steps")
    parser.add_argument("--eval-steps", type=float, default=0.10)
    parser.add_argument("--run-name", default="modernbert-base-lr-5e-5-cosine-epoch-3")
    parser.add_argument("--resume")
    parser.add_argument("--wandb-project", default="stereodetect")
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from sklearn.metrics import classification_report
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from stereodetect.models.encoders.data import Examples

    if min(args.batch_size, args.gradient_accumulation, args.max_length) < 1 or args.epochs <= 0:
        parser.error("Batch sizes, max length, and epochs must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    bf16_supported = torch.cuda.is_bf16_supported()
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model").exists() and not args.resume:
        parser.error(
            "Saved model already exists; choose another output directory or --resume a checkpoint"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train, val, test = [
        Examples(Path(args.data_dir) / f"{split}.csv", tokenizer, args.max_length)
        for split in ("train", "val", "test")
    ]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=5,
        attn_implementation="sdpa",
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )
    config = {
        **vars(args),
        "task": "five-label sentence classification",
        "input_columns": ["Sentence"],
        "label_names": LABEL_NAMES,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "precision": "bf16 autocast with fp32 parameters"
        if bf16_supported
        else "fp16 autocast with fp32 parameters",
        "model_reference": f"https://huggingface.co/{args.model}",
        "attention_implementation": "sdpa",
        "comparison_scope": "Sentence-only StereoDetect five-label task; model and hyperparameters recorded in this configuration",
        "dataset_sha256": {
            split: hashlib.sha256((Path(args.data_dir) / f"{split}.csv").read_bytes()).hexdigest()
            for split in ("train", "val", "test")
        },
        "packages": package_versions("torch", "transformers", "vllm"),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        optim=args.optim,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        report_to="wandb",
        run_name=args.run_name,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=lambda p: metrics(
            p.label_ids.tolist(), p.predictions.argmax(axis=1).tolist()
        ),
    )
    trained = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_metrics("train", trained.metrics)
    trainer.save_state()
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(output / "model")
    predicted = trainer.predict(test, metric_key_prefix="test")
    trainer.save_metrics("test", predicted.metrics)
    trainer.log(predicted.metrics)
    truth = predicted.label_ids.tolist()
    guesses = predicted.predictions.argmax(axis=1).tolist()
    logits = predicted.predictions
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    report = {
        "num_examples": len(test),
        "input_columns": ["Sentence"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        **detailed_metrics(truth, guesses),
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text_report = "Classification Report (ModernBERT, labels only):\n" + classification_report(
        truth, guesses, labels=list(range(5)), zero_division=0, digits=4
    )
    (output / "classification_report.txt").write_text(text_report)
    with (output / "test_predictions.csv").open("w", newline="") as handle:
        extra = ["prediction", "correct"] + [f"probability_{i}" for i in range(5)]
        writer = csv.DictWriter(handle, fieldnames=list(test.rows[0]) + extra)
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "prediction": guess,
                "correct": int(guess == int(row["labels"])),
                **{f"probability_{i}": float(probs[i]) for i in range(5)},
            }
            for row, guess, probs in zip(test.rows, guesses, probabilities)
        )
    print(text_report, flush=True)
    print(f"Best model saved to {output / 'model'}", flush=True)


if __name__ == "__main__":
    main()

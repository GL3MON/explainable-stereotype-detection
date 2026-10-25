"""Two-GPU, sentence-only numeric-label LoRA training on Kaggle T4s."""

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import sys
import time
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
# The Kaggle image includes an old torchao version. Plain FP32 LoRA does not
# use that optional quantization backend, so keep it out of backend discovery.
sys.modules["torchao"] = None
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils.save_and_load import get_peft_model_state_dict
from sklearn.metrics import classification_report
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from transformers.models.qwen3_5 import modeling_qwen3_5

from stereodetect.metrics import detailed_report
from stereodetect.models.qwen.data import LabelExamples
from stereodetect.models.qwen.labels import LABEL_NAMES, MODEL, SENTENCE_SYSTEM, messages

REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def collate(items, pad_id):
    """Left padding aligns the label and EOS as the final two tokens of every row."""
    length = (max(len(x["input_ids"]) for x in items) + 7) // 8 * 8
    result = {
        "input_ids": torch.tensor([
            [pad_id] * (length - len(x["input_ids"])) + x["input_ids"] for x in items
        ]),
        "attention_mask": torch.tensor([
            [0] * (length - len(x["input_ids"])) + [1] * len(x["input_ids"]) for x in items
        ]),
        "labels": torch.tensor([
            [-100] * (length - len(x["labels"])) + x["labels"] for x in items
        ]),
    }
    assert torch.all((result["labels"] != -100).sum(1) == 2)
    assert torch.all(result["labels"][:, -2:] != -100)
    return result


def compact_loss(model, batch):
    """Same masked causal-LM loss, projecting only positions needed for label + EOS."""
    output = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        logits_to_keep=3, use_cache=False,
    )
    logits = output.logits[:, :2].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"][:, -2:].reshape(-1))
    return loss, output.logits[:, 0]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    assert world == 2 and torch.cuda.device_count() == 2, "Exactly two GPUs required"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    assert "T4" in properties.name, properties.name
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    set_seed(42)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    # T4 lacks BF16. FP32 also avoids FP16 overflow in Qwen3.5's recurrent blocks.
    # Optional FLA/Hub kernels can require newer architectures; use the torch reference.
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = {s: LabelExamples(data / f"{s}.csv", tokenizer, 256, sentence_only=True)
                for s in ("train", "val", "test")}
    label_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(5)]
    assert all(len(i) == 1 for i in label_ids)
    label_ids = [i[0] for i in label_ids]
    for dataset in datasets.values():
        for row, item in zip(dataset.rows, dataset.items):
            assert len(item["labels"]) - item["labels"].count(-100) == 2
            assert item["input_ids"][-1] == tokenizer.eos_token_id
            assert item["input_ids"][-2] == label_ids[int(row["labels"])]
            assert messages(row, sentence_only=True) == messages({"Sentence": row["Sentence"]}, sentence_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
        device_map={"": device},
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model.language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, lora).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    initial_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    longest = max(datasets["train"].items, key=lambda x: len(x["input_ids"]))
    probes, chosen = [], None
    for candidate in (16, 8, 4, 2, 1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        test_optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
        ok, error = True, None
        try:
            model.train()
            test_batch = to_device(collate([longest] * candidate, tokenizer.pad_token_id), device)
            loss, _ = compact_loss(model, test_batch)
            assert torch.isfinite(loss).item(), "Non-finite probe loss"
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            test_optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            ok = peak < properties.total_memory * 0.90
        except torch.OutOfMemoryError:
            ok, error = False, "CUDA out of memory"
            peak = torch.cuda.max_memory_allocated()
        finally:
            test_optimizer.zero_grad(set_to_none=True)
            del test_optimizer
            if "test_batch" in locals():
                del test_batch
            if "loss" in locals():
                del loss
            set_peft_model_state_dict(model, initial_state)
            torch.cuda.empty_cache()
        flags = torch.tensor(int(ok), device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        record = {"rank": rank, "batch_size": candidate, "peak_allocated_bytes": peak,
                  "total_memory_bytes": properties.total_memory, "pass": ok, "error": error}
        all_records = [None] * world
        dist.all_gather_object(all_records, record)
        if rank == 0:
            probes.append(all_records)
            print("BATCH_PROBE", json.dumps(all_records), flush=True)
        if flags.item():
            chosen = candidate
            break
    assert chosen is not None, "No stable batch size found"
    del initial_state
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    set_seed(42)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False,
                                  broadcast_buffers=False, gradient_as_bucket_view=True)
    sampler = DistributedSampler(datasets["train"], num_replicas=world, rank=rank, seed=42)
    loader = DataLoader(datasets["train"], batch_size=chosen, sampler=sampler, num_workers=0,
                        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * 0.05), total_steps)
    config = {
        "model": MODEL, "model_revision": REVISION, "base_model_name_or_path": MODEL,
        "sentence_only": True, "input_columns": ["Sentence"], "output_format": "one digit 0-4",
        "enable_thinking": False, "system_prompt": SENTENCE_SYSTEM, "precision": "float32",
        "distributed": "torch DistributedDataParallel / NCCL", "world_size": world,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(world)],
        "batch_size_per_device": chosen, "gradient_accumulation_steps": 1,
        "effective_batch_size": chosen * world, "epochs": args.epochs,
        "optimizer": "AdamW", "learning_rate": 2e-4, "weight_decay": 0.01,
        "lr_scheduler": "cosine", "warmup_fraction": 0.05, "max_grad_norm": 1.0,
        "seed": 42, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "trainable_parameters": sum(p.numel() for p in params), "total_steps": total_steps,
        "train_rows": len(datasets["train"]), "validation_rows": len(datasets["val"]),
        "test_rows": len(datasets["test"]), "scored_field": "label",
        "prediction_method": "argmax over label-token logits 0-4",
        "loss": "mean causal cross-entropy over label and EOS; all prompt tokens masked",
        "kernel_backend": "PyTorch reference GDN and conv1d; SDPA full attention",
        "data_sha256": {s: hashlib.sha256((data / f"{s}.csv").read_bytes()).hexdigest() for s in datasets},
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "numpy", "scikit-learn")},
    }
    if rank == 0:
        write_json(out / "run_config.json", config)
        write_json(out / "batch_probes.json", probes)
        (out / "prompt_example.txt").write_text(tokenizer.apply_chat_template(
            messages(datasets["train"].rows[0], sentence_only=True), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ))
        print("TRAINING_CONFIG", json.dumps(config), flush=True)

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        dataset = datasets[split]
        indices = list(range(rank, len(dataset), world))
        local = []
        for start in range(0, len(indices), chosen):
            ids = indices[start:start + chosen]
            batch = to_device(collate([dataset[i] for i in ids], tokenizer.pad_token_id), device)
            _, logits = compact_loss(model, batch)
            probabilities = logits[:, label_ids].float().softmax(-1)
            predictions = probabilities.argmax(-1).tolist()
            local.extend({"index": i, "prediction": p, "probabilities": probs}
                         for i, p, probs in zip(ids, predictions, probabilities.cpu().tolist()))
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        records = sorted([r for part in gathered for r in part], key=lambda r: r["index"])
        assert [r["index"] for r in records] == list(range(len(dataset)))
        truth = [int(r["labels"]) for r in dataset.rows]
        metrics = detailed_report(truth, [r["prediction"] for r in records])
        return metrics, records

    started, step, best = time.perf_counter(), 0, -1.0
    history, best_epoch = [], None
    running_loss = torch.zeros((), device=device)
    for epoch in range(args.epochs):
        ddp.train()
        sampler.set_epoch(epoch)
        for items in loader:
            batch = to_device(items, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compact_loss(ddp, batch)
            finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            assert finite.item(), f"Non-finite loss at step {step}"
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += loss.detach()
            if step % 10 == 0:
                dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                record = {"step": step, "epoch": epoch + (step - epoch * len(loader)) / len(loader),
                          "loss": running_loss.item() / (world * 10), "grad_norm": float(grad_norm),
                          "learning_rate": scheduler.get_last_lr()[0],
                          "elapsed_seconds": time.perf_counter() - started}
                if rank == 0:
                    print("TRAIN", json.dumps(record), flush=True)
                    with (out / "history.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                running_loss.zero_()
        metrics, _ = evaluate("val")
        record = {"epoch": epoch + 1, "step": step, "validation": metrics,
                  "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        if metrics["macro_f1"] > best:
            best, best_epoch = metrics["macro_f1"], epoch + 1
            if rank == 0:
                adapter = out / "adapter"
                model.peft_config["default"].base_model_name_or_path = MODEL
                model.peft_config["default"].revision = REVISION
                model.save_pretrained(adapter)
                tokenizer.save_pretrained(adapter)
                write_json(adapter / "run_config.json", config)
        if rank == 0:
            write_json(out / "validation_history.json", history)
            print("VALIDATION", json.dumps(record), flush=True)
        dist.barrier()
    from safetensors.torch import load_file
    state = load_file(str(out / "adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, state)
    metrics, records = evaluate("test")
    if rank == 0:
        report = {
            "model": MODEL, "model_revision": REVISION, "sentence_only": True,
            "input_columns": ["Sentence"], "enable_thinking": False, "num_examples": len(records),
            "partial_evaluation": False, "best_epoch": best_epoch,
            "best_validation_macro_f1": best, "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started, "config": config,
        }
        write_json(out / "evaluation.json", report)
        write_json(out / "train_results.json", {"steps": step, "epochs": args.epochs,
                   "runtime_seconds": time.perf_counter() - started, "best_epoch": best_epoch})
        truth = [int(r["labels"]) for r in datasets["test"].rows]
        predicted = [r["prediction"] for r in records]
        text = classification_report(truth, predicted, labels=list(range(5)),
                                     target_names=LABEL_NAMES, digits=4, zero_division=0)
        (out / "classification_report.txt").write_text(text)
        fields = list(datasets["test"].rows[0]) + ["prediction", "correct"] + [f"probability_{i}" for i in range(5)]
        with (out / "test_predictions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row, record in zip(datasets["test"].rows, records):
                writer.writerow({**row, "prediction": record["prediction"],
                                 "correct": record["prediction"] == int(row["labels"]),
                                 **{f"probability_{i}": p for i, p in enumerate(record["probabilities"])}})
        write_json(out / "completion.json", {"state": "complete", "steps": step,
                   "test_examples": len(records), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        print("COMPLETE", json.dumps(report), flush=True)
        print(text, flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

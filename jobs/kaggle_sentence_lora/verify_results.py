"""Validate downloaded predictions, provenance, and the saved LoRA adapter."""

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_directory", type=Path)
    args = parser.parse_args()
    root = args.job_directory.resolve()
    out = root / "results/artifacts"
    config = json.loads((out / "run_config.json").read_text())
    report = json.loads((out / "evaluation.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    remote_manifest = json.loads((out / "input_manifest.json").read_text())
    assert manifest == remote_manifest, "Uploaded source manifest differs"
    assert hashlib.sha256((out / "train.py").read_bytes()).hexdigest() == manifest["code"]["train.py"]
    assert config["sentence_only"] and config["input_columns"] == ["Sentence"]
    assert config["enable_thinking"] is False and config["world_size"] == 2
    assert all("T4" in name for name in config["gpu_names"])
    assert config["effective_batch_size"] == 2 * config["batch_size_per_device"]
    assert config["epochs"] == 3
    for split, digest in config["data_sha256"].items():
        path = Path("data") / f"{split}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with Path("data/test.csv").open(newline="", encoding="utf-8-sig") as f:
        gold = list(csv.DictReader(f))
    with (out / "test_predictions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(gold) == report["num_examples"] == 1738
    confusion = [[0] * 6 for _ in range(5)]
    for row, source in zip(rows, gold):
        assert all(row[k] == value for k, value in source.items())
        prediction, actual = int(row["prediction"]), int(source["labels"])
        assert prediction in range(5)
        probabilities = [float(row[f"probability_{i}"]) for i in range(5)]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        assert math.isclose(sum(probabilities), 1, abs_tol=1e-5)
        assert max(range(5), key=lambda i: probabilities[i]) == prediction
        assert row["correct"] == str(prediction == actual)
        confusion[actual][prediction] += 1
    accuracy = sum(confusion[i][i] for i in range(5)) / len(rows)
    f1s, supports = [], []
    for i in range(5):
        tp = confusion[i][i]
        support = sum(confusion[i])
        count = sum(row[i] for row in confusion)
        f1s.append(2 * tp / (support + count) if support + count else 0)
        supports.append(support)
    macro_f1 = sum(f1s) / 5
    weighted_f1 = sum(f * n for f, n in zip(f1s, supports)) / len(rows)
    for name, expected in (("accuracy", accuracy), ("macro_f1", macro_f1), ("weighted_f1", weighted_f1)):
        assert math.isclose(report["metrics"][name], expected, abs_tol=1e-12), name
    assert report["metrics"]["confusion_matrix"] == confusion
    history = json.loads((out / "validation_history.json").read_text())
    assert len(history) == 3
    best = max(history, key=lambda h: h["validation"]["macro_f1"])
    assert best["epoch"] == report["best_epoch"]
    assert best["validation"]["macro_f1"] == report["best_validation_macro_f1"]
    adapter = out / "adapter/adapter_model.safetensors"
    with adapter.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        tensor_data = f.read()
    params, nonzero_b = 0, False
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        assert entry["dtype"] == "F32"
        start, end = entry["data_offsets"]
        values = np.frombuffer(tensor_data, dtype="<f4", offset=start, count=(end - start) // 4)
        assert np.isfinite(values).all(), name
        assert values.size == math.prod(entry["shape"])
        params += values.size
        if "lora_B" in name and np.any(values != 0):
            nonzero_b = True
    assert params == config["trainable_parameters"] == 16819200
    assert nonzero_b, "LoRA B matrices were never updated"
    adapter_config = json.loads((out / "adapter/adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == "Qwen/Qwen3.5-2B"
    checks = {"state": "passed", "test_examples": len(rows), "accuracy": accuracy,
              "macro_f1": macro_f1, "weighted_f1": weighted_f1,
              "finite_adapter_parameters": params,
              "sha256": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in out.rglob("*") if p.is_file()}}
    (root / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({k: v for k, v in checks.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()

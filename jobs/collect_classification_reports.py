"""Collect all saved test/validation classification reports into two text files."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

from stereodetect.data import CATEGORIES, normalize_category
from stereodetect.labels import LABEL_NAMES

INVALID = "<INVALID_OR_MISSING>"
LABEL_NAMES_WITH_IDS = [f"{i} - {name}" for i, name in enumerate(LABEL_NAMES)]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def stored_metric(report, name):
    for container in (report, report.get("metrics", {}), report.get("test_metrics", {})):
        for key in (name, "test_" + name):
            if key in container:
                return container[key]
    return None


def check_stored(report, truth, prediction):
    expected = {
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, labels=list(range(5)), average="macro", zero_division=0),
    }
    for name, value in expected.items():
        saved = stored_metric(report, name)
        if saved is not None:
            assert math.isclose(float(saved), value, abs_tol=1e-12), (name, saved, value)
    return expected


def field_report(truth, prediction, classes, names=None):
    """Macro averages use explicit gold classes; out-of-set guesses remain errors."""
    assert truth and len(truth) == len(prediction)
    result = classification_report(truth, prediction, labels=classes, target_names=names,
                                   digits=4, zero_division=0, output_dict=True)
    text = classification_report(truth, prediction, labels=classes, target_names=names,
                                 digits=4, zero_division=0)
    support_keys = names if names else classes
    assert sum(result[str(k)]["support"] for k in support_keys) == len(truth)
    accuracy = float(accuracy_score(truth, prediction))
    lines = [f"Accuracy (all {len(truth)} examples): {accuracy:.6f} ({accuracy:.2%})", text.rstrip()]
    extras = Counter(p for p in prediction if p not in set(classes))
    if extras:
        lines += ["", "Predicted values outside the scored gold-class set (all counted as errors):",
                  "Each has gold support 0 and precision/recall/F1 0; excluded from macro/weighted averages."]
        for value, count in sorted(extras.items(), key=lambda item: (-item[1], str(item[0]))):
            lines.append(f"  {json.dumps(value, ensure_ascii=False)}: predicted {count} times")
    return "\n".join(lines) + "\n", {
        "accuracy": accuracy, "macro_f1": result["macro avg"]["f1-score"],
        "weighted_f1": result["weighted avg"]["f1-score"],
        "outside_gold_class_predictions": sum(extras.values()),
        "scored_classes": len(classes), "support": len(truth),
    }


def section(title):
    return "\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100 + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {s: read_csv(Path("data") / f"{s}.csv") for s in ("test", "val")}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "label_only": [], "json": []}

    label_runs = [
        ("Qwen3.5-2B BASE - sentence only -> label", "outputs/eval/qwen35_2b_base_sentence_labels_nothinking_20261022_195718", "predictions.csv", "Sentence only"),
        ("Qwen3.5-2B LoRA - sentence only -> label (Kaggle 2xT4)", "outputs/kaggle_qwen35_sentence_lora_1022-183119/results/artifacts", "test_predictions.csv", "Sentence only"),
        ("Qwen3.5-2B BASE - category + target + sentence -> label", "outputs/eval/qwen35_2b_base_nothinking_20261022_173617/labels", "predictions.csv", "Category + Target + Sentence"),
        ("Qwen3.5-2B LoRA - conditioned label, training-time test evaluation", "outputs/qwen35_2b_stereodetect", "test_predictions.csv", "Category + Target + Sentence"),
        ("Qwen3.5-2B LoRA - conditioned label, standalone test evaluation", "outputs/eval/adapter_test", "predictions.csv", "Category + Target + Sentence"),
        ("ALBERT-base-v2 fine-tuned", "outputs/albert_stereodetect", "test_predictions.csv", "Sentence only"),
        ("ModernBERT-base fine-tuned - original run", "outputs/modernbert_stereodetect", "test_predictions.csv", "Sentence only"),
        ("ModernBERT-base fine-tuned - LR 5e-5, cosine, 3 epochs, seed 3407", "outputs/modernbert_stereodetect_lr5e5_cosine_ep3_seed3407", "test_predictions.csv", "Sentence only"),
        ("ModernBERT-large fine-tuned - LR 3e-5, warmup 30, WD 0.01, 3 epochs, seed 3407", "outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407", "test_predictions.csv", "Sentence only"),
    ]
    label_parts, label_summary = [], []
    for number, (title, folder, name, inputs) in enumerate(label_runs, 1):
        folder = Path(folder)
        rows = read_csv(folder / name)
        assert len(rows) == len(datasets["test"]) == 1738
        for row, gold in zip(rows, datasets["test"]):
            assert all(row[k] == gold[k] for k in ("Sentence", "Category", "Target", "labels"))
        truth = [int(r["labels"]) for r in rows]
        prediction = [int(r["prediction"]) for r in rows]
        saved = read_json(folder / "evaluation.json")
        check_stored(saved, truth, prediction)
        report, metrics = field_report(truth, prediction, list(range(5)), LABEL_NAMES_WITH_IDS)
        source = folder / name
        label_summary.append((number, title, metrics["accuracy"], metrics["macro_f1"]))
        cfg_path = folder / "run_config.json"
        cfg = read_json(cfg_path) if cfg_path.exists() else {}
        cfg = {**cfg, **saved.get("config", {})}
        thinking = saved.get("enable_thinking", cfg.get("enable_thinking", "N/A for encoder models"))
        extra = []
        if "engine" in cfg:
            extra.append(f"Runtime: vLLM; dtype={cfg['engine'].get('dtype')}")
        if "precision" in cfg:
            extra.append(f"Runtime precision: {cfg['precision']}")
        if saved.get("best_epoch"):
            extra.append(f"Selected epoch: {saved['best_epoch']}")
        label_parts += [section(f"TEST {number:02d}. {title}"),
                        f"Split: TEST, complete, 1738 / 1738 examples\nInput: {inputs}\nOutput: label 0-4\nThinking: {thinking}\n",
                        "\n".join(extra) + "\n" if extra else "",
                        f"Prediction source: {source}\nSaved report: {folder / 'evaluation.json'}\n\n", report]
        manifest["label_only"].append({"title": title, "source": str(source), "split": "test",
                                        "num_examples": len(rows), **metrics})

    label_parts += [section("APPENDIX: AVAILABLE VALIDATION CLASSIFICATION REPORTS"),
                    "These are validation results used for model selection, not held-out test results.\n"]
    history_path = Path("outputs/kaggle_qwen35_sentence_lora_1022-183119/results/artifacts/validation_history.json")
    for entry in read_json(history_path):
        matrix = entry["validation"]["confusion_matrix"]
        truth, prediction = [], []
        for actual, row in enumerate(matrix):
            for guess, count in enumerate(row):
                truth.extend([actual] * count)
                prediction.extend([guess if guess < 5 else -1] * count)
        assert len(truth) == 868
        check_stored(entry["validation"], truth, prediction)
        report, metrics = field_report(truth, prediction, list(range(5)), LABEL_NAMES_WITH_IDS)
        title = f"Kaggle sentence-only LoRA - validation epoch {entry['epoch']} (step {entry['step']})"
        label_parts += [section(title), f"Split: VALIDATION, complete, 868 / 868 examples\nSource: {history_path}\nReconstructed exactly from the saved validation confusion matrix.\n\n", report]
        manifest["label_only"].append({"title": title, "source": str(history_path), "split": "val",
                                        "num_examples": len(truth), **metrics})

    label_header = ["ALL LABEL-ONLY CLASSIFICATION REPORTS\n",
                    f"Generated UTC: {manifest['created_utc']}\n",
                    "Includes all nine completed label-only test evaluations: Qwen, ALBERT and ModernBERT.\n",
                    "JSON-output experiments are in the separate category/label/target report.\n",
                    "All full test reports use the same 1,738 rows. Metrics were recomputed and checked against saved evaluations.\n",
                    "Category/target-conditioned and sentence-only inputs differ; inspect each run's input contract.\n",
                    "The base sentence-only run used vLLM BF16; the Kaggle LoRA run used PyTorch FP32.\n",
                    "Precision, recall and F1 are fractions; the accuracy line also gives a percentage.\n",
                    "Labels: " + "; ".join(LABEL_NAMES_WITH_IDS) + ".\n",
                    "All label reports include labels 0-4 in macro averages; invalid predictions remain errors.\n",
                    "\nCOMPLETED TEST RUN SUMMARY\n",
                    f"{'ID':<4}{'Accuracy':>12}{'Macro F1':>12}  Experiment\n"]
    for number, title, accuracy, macro in label_summary:
        label_header.append(f"{number:<4}{accuracy:>12.4%}{macro:>12.4%}  {title}\n")
    label_footer = section("COVERAGE NOTES") + (
        "The stopped MIG 0 sentence-only training has no completed final test evaluation and is excluded.\n"
        "Training smoke checks without classification predictions are excluded.\n"
        "Training-time and standalone evaluations are both retained: their saved prediction vectors differ.\n"
        "Encoder validation aggregate scores without per-class predictions/confusion matrices cannot yield full reports.\n"
        "No model inference or training was run to create these text files.\n"
    )
    label_path = args.output_dir / "all_label_only_classification_reports.txt"
    label_path.write_text("".join(label_header + label_parts) + label_footer)

    json_runs = [
        ("Qwen3.5-2B BASE - sentence -> JSON", "outputs/eval/qwen35_2b_base_nothinking_20261022_173617/json/completions.jsonl", "test", "COMPLETE TEST"),
        ("Qwen3.5-2B JSON LoRA - training-time test evaluation", "outputs/qwen35_2b_stereodetect_json/test_completions.jsonl", "test", "COMPLETE TEST"),
        ("Qwen3.5-2B JSON LoRA - standalone test evaluation", "outputs/eval/json_test/completions.jsonl", "test", "COMPLETE TEST"),
    ]
    for step in (380, 760, 1140):
        json_runs.append((f"Qwen3.5-2B JSON LoRA - validation epoch {step // 380}",
                          f"outputs/qwen35_2b_stereodetect_json/validation_completions_step_{step}.jsonl", "val", "COMPLETE VALIDATION"))
    json_runs += [
        ("JSON LoRA vLLM thinking - unfinished saved test subset", "outputs/eval/json_test_vllm_thinking_limited_20261014/completions.jsonl", "test", "INCOMPLETE TEST SUBSET"),
        ("JSON LoRA vLLM thinking - test smoke", "outputs/eval/json_test_vllm_thinking_limited_20261014/smoke/completions.jsonl", "test", "PARTIAL TEST SMOKE"),
        ("JSON LoRA vLLM thinking - sampled test smoke", "outputs/eval/json_test_vllm_thinking_limited_20261014/smoke_sampled/completions.jsonl", "test", "PARTIAL TEST SMOKE"),
        ("JSON LoRA vLLM thinking - validation budget 256 smoke", "outputs/eval/json_val_vllm_budget_smoke_20261014/budget_256/completions.jsonl", "val", "PARTIAL VALIDATION SMOKE"),
        ("JSON LoRA vLLM thinking - validation budget 512 smoke", "outputs/eval/json_val_vllm_budget_smoke_20261014/budget_512/completions.jsonl", "val", "PARTIAL VALIDATION SMOKE"),
        ("JSON LoRA vLLM thinking - final-JSON validation budget 256 smoke", "outputs/eval/json_val_vllm_final_json_smoke_20261014/budget_256/completions.jsonl", "val", "PARTIAL VALIDATION SMOKE"),
    ]
    json_parts, summary = [], []
    for number, (title, source, split, coverage) in enumerate(json_runs, 1):
        source = Path(source)
        records = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        if records and "index" in records[0]:
            records.sort(key=lambda r: r["index"])
            indices = [r["index"] for r in records]
            assert len(set(indices)) == len(indices)
        else:
            indices = list(range(len(records)))
        assert all(0 <= i < len(datasets[split]) for i in indices)
        if coverage.startswith("COMPLETE"):
            assert indices == list(range(len(datasets[split])))
        gold_rows = [datasets[split][i] for i in indices]
        truth = {"category": [], "label": [], "target": []}
        predicted = {"category": [], "label": [], "target": []}
        parse_failures = invalid_category = invalid_target = invalid_label = 0
        for record, gold in zip(records, gold_rows):
            if "Sentence" in record:
                assert all(record[k] == gold[k] for k in ("Sentence", "Category", "Target", "labels"))
            else:
                assert record["gold_json"] == {"category": normalize_category(gold["Category"]),
                                               "target": gold["Target"], "label": int(gold["labels"])}
            completion = record.get("completion", record.get("final_answer", ""))
            try:
                obj = json.loads(completion)
                if not isinstance(obj, dict):
                    raise ValueError("JSON object required")
            except (TypeError, ValueError):
                obj = {}
                parse_failures += 1
            category, target, label = (obj.get(key) for key in ("category", "target", "label"))
            category_ok = isinstance(category, str) and bool(category.strip())
            target_ok = isinstance(target, str) and bool(target.strip())
            label_ok = type(label) is int and label in range(5)
            invalid_category += not category_ok
            invalid_target += not target_ok
            invalid_label += not label_ok
            truth["category"].append(normalize_category(gold["Category"]))
            truth["target"].append(gold["Target"])
            truth["label"].append(int(gold["labels"]))
            predicted["category"].append(normalize_category(category) if category_ok else INVALID)
            predicted["target"].append(target if target_ok else INVALID)
            predicted["label"].append(label if label_ok else -1)
            saved_prediction = record.get("prediction", record.get("predicted_label"))
            if saved_prediction is not None:
                assert saved_prediction == predicted["label"][-1]
        evaluation_path = source.parent / "evaluation.json"
        if "validation_completions" not in source.name and evaluation_path.exists():
            check_stored(read_json(evaluation_path), truth["label"], predicted["label"])
        config_path = source.parent / "run_config.json"
        cfg = read_json(config_path) if config_path.exists() else {}
        saved = read_json(evaluation_path) if evaluation_path.exists() else {}
        cfg = {**cfg, **saved}
        thinking = cfg.get("enable_thinking", False)
        fields = {}
        texts = {}
        for field in ("category", "label", "target"):
            classes = CATEGORIES if field == "category" else list(range(5)) if field == "label" else sorted(set(truth[field]))
            names = LABEL_NAMES_WITH_IDS if field == "label" else None
            texts[field], fields[field] = field_report(truth[field], predicted[field], classes, names)
        joint = sum(all(truth[key][i] == predicted[key][i] for key in truth) for i in range(len(records)))
        json_parts += [section(f"JSON {number:02d}. {title}"),
                       f"Coverage: {coverage}\nSplit: {split.upper()}; {len(records)} / {len(datasets[split])} saved examples\n",
                       "Input: Sentence only\nOutput: category, target, label JSON\n",
                       f"Thinking enabled: {thinking}\nSource: {source}\nGold CSV: data/{split}.csv\n",
                       f"JSON parse/object failures: {parse_failures}\nInvalid/missing category strings: {invalid_category}\n",
                       f"Invalid/missing target strings: {invalid_target}\nInvalid/missing labels: {invalid_label}\n",
                       f"All three fields correct: {joint}/{len(records)} ({joint / len(records):.4%})\n"]
        for key in ("max_tokens", "thinking_token_budget", "max_new_tokens", "temperature"):
            if key in cfg:
                json_parts.append(f"{key}: {cfg[key]}\n")
        for field in ("category", "label", "target"):
            json_parts += ["\n" + "-" * 100 + "\n" + field.upper() + " CLASSIFICATION REPORT\n" + "-" * 100 + "\n", texts[field]]
        item = {"title": title, "source": str(source), "split": split, "coverage": coverage,
                "num_examples": len(records), "dataset_examples": len(datasets[split]),
                "thinking_enabled": thinking, "fields": fields,
                "all_fields_accuracy": joint / len(records), "parse_failures": parse_failures}
        manifest["json"].append(item)
        summary.append((number, item))
        # Independently cross-check the existing category/target field audit.
        audit_path = source.parent / "field_validation.json"
        if audit_path.exists() and source.name == "completions.jsonl":
            audit = read_json(audit_path)
            for field in ("category", "target"):
                assert math.isclose(audit[field]["accuracy"], fields[field]["accuracy"], abs_tol=1e-12)

    json_header = ["ALL CATEGORY, LABEL AND TARGET CLASSIFICATION REPORTS (JSON-OUTPUT EXPERIMENTS)\n",
                   f"Generated UTC: {manifest['created_utc']}\n",
                   "Three completed TEST runs appear first, followed by three full VALIDATION evaluations.\n",
                   "The final six entries are clearly marked incomplete/partial diagnostic runs, not full-test results.\n",
                   "Each experiment has a separate CATEGORY, LABEL and TARGET precision/recall/F1/support report.\n",
                   "\nSCORING RULES\n",
                   "Category: trim, case-fold and collapse whitespace on gold and predicted category; fixed five categories.\n",
                   "Target: exact string match, preserving capitalization, underscores, spelling and whitespace.\n",
                   "Label: strict integer 0-4; booleans, strings, missing labels and malformed JSON are invalid.\n",
                   "Only the saved final completion is parsed; reasoning is never used to recover a prediction.\n",
                   "Malformed/missing fields count as errors; no examples are dropped.\n",
                   "Macro/weighted averages use the fixed five category/label classes, or gold target classes present in that run.\n",
                   "Predicted-only/invalid class values are listed separately and remain incorrect predictions.\n",
                   "Accuracy always includes all saved examples. Supports sum to the saved-example count in every report.\n",
                   "All per-class values are fractions; accuracy lines also give percentages.\n",
                   "Labels: " + "; ".join(LABEL_NAMES_WITH_IDS) + ".\n",
                   "\nRUN SUMMARY\n",
                   f"{'ID':<4}{'N':>6}{'Category acc':>15}{'Label acc':>13}{'Target acc':>13}  Coverage / experiment\n"]
    for number, item in summary:
        f = item["fields"]
        json_header.append(f"{number:<4}{item['num_examples']:>6}{f['category']['accuracy']:>15.4%}{f['label']['accuracy']:>13.4%}{f['target']['accuracy']:>13.4%}  {item['coverage']} - {item['title']}\n")
    json_footer = section("COVERAGE AND COMPARISON NOTES") + (
        "The 410-example thinking benchmark is unfinished and does not estimate the full test result.\n"
        "Validation results are model-selection results, not held-out test results. Smoke subsets are diagnostic only.\n"
        "The empty outputs/eval/thinking_check/completions.jsonl contains no scorable predictions.\n"
        "Training-time and standalone test reports are both included because their saved completions differ.\n"
        "Category/target reports were newly calculated from saved JSON; no new model generations were requested.\n"
    )
    json_path = args.output_dir / "all_category_label_target_classification_reports.txt"
    json_path.write_text("".join(json_header + json_parts) + json_footer)
    manifest["outputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (label_path, json_path)
    }
    (args.output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"label_only": {"test_reports": len(label_runs), "validation_reports": 3},
                      "json": {"complete_test": 3, "complete_validation": 3, "partial_diagnostics": 6},
                      "files": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()

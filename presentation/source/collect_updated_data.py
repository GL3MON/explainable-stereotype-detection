"""Collect the updated full-test reports and two existing attribution examples."""

import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "deck_data.json").read_text())
manifest_path = ROOT / "outputs/reports/classification_reports_20261022/report_manifest.json"
manifest = json.loads(manifest_path.read_text())


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


gold = csv_rows(ROOT / "data/test.csv")
data["label_runs"] = []
for entry in manifest["label_only"]:
    if entry["split"] != "test":
        continue
    rows = csv_rows(ROOT / entry["source"])
    assert len(rows) == len(gold) == 1738
    assert all(all(row[k] == g[k] for k in g) for row, g in zip(rows, gold))
    report = classification_report([int(r["labels"]) for r in rows],
                                   [int(r["prediction"]) for r in rows],
                                   labels=list(range(5)), output_dict=True, zero_division=0)
    data["label_runs"].append({**entry, "classification_report": report})

for key, source in {
    "base_sentence_labels": "outputs/eval/qwen35_2b_base_sentence_labels_nothinking_20261022_195718/evaluation.json",
    "ft_sentence_labels": "outputs/kaggle_qwen35_sentence_lora_1022-183119/results/artifacts/evaluation.json",
}.items():
    value = json.loads((ROOT / source).read_text())
    data["sources"][key] = source
    data["results"][key] = value["metrics"]
    if key.startswith("ft_"):
        data["sentence_lora_config"] = value["config"]

data["json_runs"] = []
for entry in manifest["json"]:
    if entry["coverage"] != "COMPLETE TEST":
        continue
    records = [json.loads(s) for s in (ROOT / entry["source"]).read_text().splitlines()]
    records.sort(key=lambda r: r.get("index", 0))
    assert all(all(row[k] == g[k] for k in g) for row, g in zip(records, gold))
    objects = [json.loads(r["completion"]) for r in records]
    target_gold = [g["Target"] for g in gold]
    reports = {}
    for field in ("category", "target", "label"):
        normalize = lambda value: " ".join(value.strip().casefold().split())
        truth = [normalize(g["Category"]) for g in gold] if field == "category" else target_gold if field == "target" else [int(g["labels"]) for g in gold]
        prediction = [normalize(o[field]) for o in objects] if field == "category" else [o[field] for o in objects]
        labels = ["gender", "race", "profession", "religion", "neutral"] if field == "category" else sorted(set(target_gold)) if field == "target" else list(range(5))
        report = classification_report(truth, prediction, labels=labels, output_dict=True, zero_division=0)
        assert abs(report["macro avg"]["f1-score"] - entry["fields"][field]["macro_f1"]) < 1e-12
        reports[field] = report
    data["json_runs"].append({**entry, "classification_reports": reports})

explanations = [json.loads(s) for s in (ROOT / data["explanation_source"]).read_text().splitlines()]
data["explanation_examples"] = [next(r for r in explanations if r["test_index"] == i) for i in (1713, 1205)]
for record in data["explanation_examples"]:
    g = gold[record["test_index"]]
    assert record["sentence"] == g["Sentence"] and record["gold_label"] == int(g["labels"])
    detail = record["explanations"][str(record["predicted_label"])]
    assert len(record["words"]) == len(detail["lime_values"]) == len(detail["shap_values"])

data["report_sources"] = ["outputs/reports/classification_reports_20261022/all_label_only_classification_reports.txt",
                          "outputs/reports/classification_reports_20261022/all_category_label_target_classification_reports.txt"]
data["report_manifest"] = str(manifest_path.relative_to(ROOT))
data["target_classes"] = sorted(set(g["Target"] for g in gold))
data["target_support"] = dict(Counter(g["Target"] for g in gold))
data["partial_json_runs"] = [{k: entry[k] for k in ("title", "coverage", "num_examples", "fields")}
                             for entry in manifest["json"] if entry["coverage"] != "COMPLETE TEST"]
(HERE / "deck_data_updated.json").write_text(json.dumps(data, indent=2) + "\n")
print("Collected", len(data["label_runs"]), "label-only test runs,", len(data["json_runs"]),
      "JSON test runs,", len(data["target_classes"]), "gold target classes, and two attribution examples.")

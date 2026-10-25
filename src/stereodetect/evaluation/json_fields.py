"""Validate saved JSON category/target predictions against dataset annotations."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from stereodetect.data import normalize_category


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", default="outputs/eval/json_test/completions.jsonl")
    parser.add_argument("--csv", default="data/test.csv")
    parser.add_argument("--output-dir", default="outputs/eval/json_test")
    args = parser.parse_args(argv)
    with open(args.csv, newline="", encoding="utf-8-sig") as handle:
        gold_rows = list(csv.DictReader(handle))
    with open(args.completions, encoding="utf-8") as handle:
        saved = [json.loads(line) for line in handle if line.strip()]
    if not gold_rows or len(saved) != len(gold_rows):
        parser.error("Expected one saved completion per dataset example")
    checks = []
    parse_failures = 0
    for index, (gold, record) in enumerate(zip(gold_rows, saved)):
        if any(
            gold[key] != record.get(key) for key in ("Category", "Target", "Sentence", "labels")
        ):
            parser.error(
                f"Completion {index} does not match the dataset row; refusing to score misaligned data"
            )
        try:
            prediction = json.loads(record["completion"])
            if not isinstance(prediction, dict):
                raise ValueError("JSON must be an object")
        except (json.JSONDecodeError, TypeError, ValueError):
            prediction = {}
            parse_failures += 1
        category = prediction.get("category")
        target = prediction.get("target")
        label = prediction.get("label")
        category_valid = isinstance(category, str) and bool(category.strip())
        target_valid = isinstance(target, str) and bool(target.strip())
        category_gold = normalize_category(gold["Category"])
        category_predicted = normalize_category(category) if category_valid else "<invalid>"
        category_correct = category_valid and category_predicted == category_gold
        target_correct = target_valid and target == gold["Target"]
        label_correct = type(label) is int and label == int(gold["labels"])
        checks.append(
            {
                "index": index,
                "sentence": gold["Sentence"],
                "gold_category": gold["Category"],
                "normalized_gold_category": category_gold,
                "predicted_category": category if category_valid else "",
                "normalized_predicted_category": category_predicted,
                "gold_target": gold["Target"],
                "predicted_target": target if target_valid else "",
                "gold_label": int(gold["labels"]),
                "predicted_label": label,
                "category_correct": category_correct,
                "target_correct": target_correct,
                "category_and_target_correct": category_correct and target_correct,
                "label_correct": label_correct,
                "all_fields_correct": category_correct and target_correct and label_correct,
                "category_valid": category_valid,
                "target_valid": target_valid,
                "completion": record["completion"],
            }
        )
    count = len(checks)

    def score(key):
        correct = sum(row[key] for row in checks)
        return {"correct": correct, "incorrect": count - correct, "accuracy": correct / count}

    categories = sorted({r["normalized_gold_category"] for r in checks})
    predicted_categories = sorted(
        set(categories) | {r["normalized_predicted_category"] for r in checks}
    )
    confusion = [
        [
            sum(
                r["normalized_gold_category"] == gold
                and r["normalized_predicted_category"] == predicted
                for r in checks
            )
            for predicted in predicted_categories
        ]
        for gold in categories
    ]
    per_category = {}
    for category in categories:
        subset = [r for r in checks if r["normalized_gold_category"] == category]
        per_category[category] = {
            "support": len(subset),
            "category_accuracy": sum(r["category_correct"] for r in subset) / len(subset),
            "target_accuracy": sum(r["target_correct"] for r in subset) / len(subset),
        }
    target_mismatches = Counter(
        (r["gold_target"], r["predicted_target"]) for r in checks if not r["target_correct"]
    )
    report = {
        "dataset": args.csv,
        "completions": args.completions,
        "num_examples": count,
        "category_comparison": "strip, casefold, collapse whitespace on both gold and predicted category",
        "target_comparison": "exact string equality; capitalization, underscores, and whitespace preserved",
        "category": score("category_correct"),
        "target": score("target_correct"),
        "category_and_target": score("category_and_target_correct"),
        "all_fields": score("all_fields_correct"),
        "json_parse_or_object_failures": parse_failures,
        "invalid_category_fields": sum(not r["category_valid"] for r in checks),
        "invalid_target_fields": sum(not r["target_valid"] for r in checks),
        "category_confusion_matrix": {
            "rows": categories,
            "columns": predicted_categories,
            "values": confusion,
        },
        "per_category": per_category,
        "target_mismatch_pairs": [
            {"gold": gold, "predicted": predicted, "count": n}
            for (gold, predicted), n in target_mismatches.most_common()
        ],
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "field_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    for name, rows in [
        ("field_validation.csv", checks),
        (
            "field_mismatches.csv",
            [r for r in checks if not (r["category_correct"] and r["target_correct"])],
        ),
    ]:
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        f"Category and Target Validation ({count} examples)",
        "Category: normalized case and whitespace; target: exact match.",
        "",
    ]
    for name in ("category", "target", "category_and_target", "all_fields"):
        result = report[name]
        lines.append(
            f"{name}: {result['accuracy']:.4%} ({result['correct']}/{count}); incorrect={result['incorrect']}"
        )
    lines.append(f"JSON parse/object failures: {parse_failures}")
    lines.append(f"Invalid category fields: {report['invalid_category_fields']}")
    lines.append(f"Invalid target fields: {report['invalid_target_fields']}")
    text = "\n".join(lines) + "\n"
    (output / "field_validation.txt").write_text(text)
    assert sum(map(sum, confusion)) == count
    assert report["all_fields"]["correct"] <= report["category_and_target"]["correct"]
    assert report["category_and_target"]["correct"] <= min(
        report["category"]["correct"], report["target"]["correct"]
    )
    print(text)
    print(f"Report: {output / 'field_validation.txt'}")
    print(f"Mismatches: {output / 'field_mismatches.csv'}")


if __name__ == "__main__":
    main()

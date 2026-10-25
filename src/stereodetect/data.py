"""Dataset readers preserving each task's input contract."""

import csv
from pathlib import Path

from stereodetect.labels import LABELS

CATEGORIES = ["gender", "race", "profession", "religion", "neutral"]


def normalize_category(value):
    return " ".join(value.strip().casefold().split())


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"Sentence", "labels"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: Sentence and labels columns required")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no examples")
    for index, row in enumerate(rows, 2):
        if not row["Sentence"].strip() or int(row["labels"]) not in range(5):
            raise ValueError(f"{path}:{index}: invalid sentence or label")
    return rows


def load_json_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Category", "Target", "Sentence", "labels"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no examples")
    for index, row in enumerate(rows, 2):
        if None in row or any(row.get(k) is None for k in required):
            raise ValueError(f"{path}:{index}: malformed CSV row")
        if not row["Sentence"].strip() or not row["Target"].strip():
            raise ValueError(f"{path}:{index}: empty sentence or target")
        if normalize_category(row["Category"]) not in CATEGORIES or int(row["labels"]) not in range(
            5
        ):
            raise ValueError(f"{path}:{index}: invalid category or label")
    return rows


def load_label_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"Category", "Target", "Sentence", "labels"}
        if not required.issubset(fields):
            raise ValueError(f"{path}: missing CSV columns {sorted(required - set(fields))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV contains no examples")
    for line, row in enumerate(rows, start=2):
        try:
            label = int(row["labels"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line}: labels must be integers from 0 to 4") from error
        if label not in LABELS or not (row["Sentence"] or "").strip():
            raise ValueError(f"{path}:{line}: invalid label or empty sentence")
        if any(row.get(field) is None for field in required):
            raise ValueError(f"{path}:{line}: missing CSV values")
    return rows, fields

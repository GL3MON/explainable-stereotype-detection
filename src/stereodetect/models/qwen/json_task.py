"""Sentence-only structured generation task."""

import json

from stereodetect.data import CATEGORIES, normalize_category
from stereodetect.data import load_json_rows as load_rows
from stereodetect.labels import LABEL_NAMES
from stereodetect.metrics import detailed_report, label_metrics
from stereodetect.models.qwen.labels import MODEL

__all__ = [
    "MODEL",
    "SYSTEM",
    "CATEGORIES",
    "LABEL_NAMES",
    "messages",
    "answer",
    "normalize_category",
    "load_rows",
    "parse_label",
    "label_metrics",
    "detailed_report",
]
SYSTEM = (
    "Identify the category, target group, and classification label of the sentence. "
    'Return exactly one JSON object with keys "category", "target", and "label". '
    "Category must be one of gender, race, profession, religion, neutral. "
    "Preserve the dataset target naming, spelling, capitalization, and underscores; use Neutral when there is no target. "
    "Label must be an integer: 0 = Anti-stereotype; 1 = Stereotype; "
    "2 = Neutral without target term; 3 = Neutral with target term; 4 = Bias. "
    "Do not include reasoning, Markdown, or any text outside the JSON object."
)


def messages(row):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sentence: {row['Sentence']}"},
    ]


def answer(row):
    return {
        "category": normalize_category(row["Category"]),
        "target": row["Target"],
        "label": int(row["labels"]),
    }


def parse_label(completion):
    try:
        obj = json.loads(completion.strip())
    except (json.JSONDecodeError, TypeError):
        return -1
    if not isinstance(obj, dict):
        return -1
    label = obj.get("label")
    # Category/target correctness and presence do not affect the label score.
    return label if type(label) is int and label in range(5) else -1

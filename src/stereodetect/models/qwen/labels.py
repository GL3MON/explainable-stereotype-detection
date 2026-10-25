"""Category/target-conditioned label task."""

import json
from pathlib import Path

from stereodetect.labels import LABEL_NAMES

__all__ = ["MODEL", "SYSTEM", "LABEL_NAMES", "messages"]

MODEL = "Qwen/Qwen3.5-2B"
SYSTEM = (
    "Classify the sentence using the provided category and target. "
    "Labels: 0 = Anti-stereotype; 1 = Stereotype; "
    "2 = Neutral without target term; 3 = Neutral with target term; 4 = Bias. "
    "Respond with exactly one digit from 0 to 4. Do not include reasoning or explanation."
)


SENTENCE_SYSTEM = SYSTEM.replace(
    "Classify the sentence using the provided category and target.",
    "Classify the sentence.",
)


def sentence_only_for_adapter(adapter):
    """Recover the input contract stored with a trained adapter or checkpoint."""
    if adapter is None:
        return False
    root = Path(adapter)
    for path in (root / "run_config.json", root.parent / "run_config.json"):
        if path.is_file():
            return bool(json.loads(path.read_text()).get("sentence_only", False))
    return False


def messages(row, sentence_only=False):
    if sentence_only:
        return [
            {"role": "system", "content": SENTENCE_SYSTEM},
            {"role": "user", "content": f"Sentence: {row['Sentence']}"},
        ]
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Category: {row['Category']}\nTarget: {row['Target']}\nSentence: {row['Sentence']}",
        },
    ]

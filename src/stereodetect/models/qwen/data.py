"""Supervised Qwen datasets and completion-only padding."""

import csv
import json

from stereodetect.models.qwen.json_task import answer, load_rows
from stereodetect.models.qwen.json_task import messages as json_messages
from stereodetect.models.qwen.labels import messages as label_messages


class LabelExamples:
    def __init__(self, path, tokenizer, max_length, sentence_only=False):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        self.items = []
        for row in self.rows:
            label = int(row["labels"])
            if label not in range(5) or not row["Sentence"].strip():
                raise ValueError(f"Invalid row in {path}: {row}")
            prompt = tokenizer.apply_chat_template(
                label_messages(row, sentence_only=sentence_only),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            completion = tokenizer.encode(str(label) + "<|im_end|>", add_special_tokens=False)
            if len(prompt) + len(completion) > max_length:
                raise ValueError(
                    f"Example has {len(prompt) + len(completion)} tokens, above {max_length}; increase --max-length"
                )
            self.items.append(
                {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion}
            )
        print(
            f"{path}: {len(self.items)} examples; longest {max(len(x['input_ids']) for x in self.items)} tokens",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class JsonExamples:
    def __init__(self, path, tokenizer, max_length):
        self.rows = load_rows(path)
        self.items = []
        for row in self.rows:
            prompt = tokenizer.apply_chat_template(
                json_messages(row),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            text = json.dumps(answer(row), ensure_ascii=False) + "<|im_end|>"
            completion = tokenizer.encode(text, add_special_tokens=False)
            if len(prompt) + len(completion) > max_length:
                raise ValueError(
                    f"{path}: example exceeds {max_length} tokens; increase --max-length"
                )
            self.items.append(
                {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion}
            )
        print(
            f"{path}: {len(self.items)} examples; longest {max(len(x['input_ids']) for x in self.items)} tokens",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        import torch

        length = ((max(len(x["input_ids"]) for x in items) + 7) // 8) * 8
        return {
            "input_ids": torch.tensor(
                [x["input_ids"] + [self.pad_id] * (length - len(x["input_ids"])) for x in items]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(x["input_ids"]) + [0] * (length - len(x["input_ids"])) for x in items]
            ),
            "labels": torch.tensor(
                [x["labels"] + [-100] * (length - len(x["labels"])) for x in items]
            ),
        }

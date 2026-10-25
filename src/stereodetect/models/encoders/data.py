"""Sentence-only encoder dataset."""

from stereodetect.data import read_rows


class Examples:
    def __init__(self, path, tokenizer, max_length):
        self.rows = read_rows(path)
        encoded = tokenizer([row["Sentence"] for row in self.rows], truncation=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if max(lengths) > max_length:
            raise ValueError(
                f"{path}: longest example is {max(lengths)} tokens; increase --max-length"
            )
        self.items = [
            {
                **{key: values[index] for key, values in encoded.items()},
                "labels": int(row["labels"]),
            }
            for index, row in enumerate(self.rows)
        ]
        print(f"{path}: {len(self.items)} examples, longest {max(lengths)} tokens", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

"""Transformers trainer for free-generation JSON validation (training extra)."""

import json
from pathlib import Path

import torch
from transformers import Trainer

from stereodetect.models.qwen.json_task import parse_label


class JsonTrainer(Trainer):
    """Evaluate free JSON generations; never supply the gold JSON as generation input."""

    def __init__(self, *args, max_new_tokens=96, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.completions = []

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        tokenizer = self.processing_class
        prompts, gold_texts = [], []
        for ids, labels in zip(inputs["input_ids"], inputs["labels"]):
            first = int((labels != -100).nonzero()[0])
            prompts.append(ids[:first].tolist())
            gold_texts.append(
                tokenizer.decode(labels[labels != -100].tolist(), skip_special_tokens=True)
            )
        length = max(map(len, prompts))
        generation_inputs = {
            "input_ids": torch.tensor(
                [[tokenizer.pad_token_id] * (length - len(p)) + p for p in prompts],
                device=inputs["input_ids"].device,
            ),
            "attention_mask": torch.tensor(
                [[0] * (length - len(p)) + [1] * len(p) for p in prompts],
                device=inputs["input_ids"].device,
            ),
        }
        generation_model = self.accelerator.unwrap_model(model)
        with torch.no_grad(), self.compute_loss_context_manager():
            generated = generation_model.generate(
                **generation_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, length:], skip_special_tokens=True)
        actual, predicted = [], []
        for gold_text, text in zip(gold_texts, texts):
            gold = json.loads(gold_text)
            guess = parse_label(text)
            actual.append(gold["label"])
            predicted.append(guess)
            self.completions.append(
                {
                    "gold_json": gold,
                    "completion": text,
                    "predicted_label": guess,
                    "correct": guess == gold["label"],
                }
            )
        device = inputs["input_ids"].device
        return (
            None,
            torch.tensor(predicted, device=device).reshape(-1, 1),
            torch.tensor(actual, device=device).reshape(-1, 1),
        )

    def evaluate(self, *args, **kwargs):
        self.completions = []
        result = super().evaluate(*args, **kwargs)
        path = (
            Path(self.args.output_dir)
            / f"validation_completions_step_{self.state.global_step}.jsonl"
        )
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.completions)
        )
        return result

    def predict(self, *args, **kwargs):
        self.completions = []
        return super().predict(*args, **kwargs)

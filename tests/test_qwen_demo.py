"""Check demo sampling, adapter routing, and strict output scoring without a GPU."""

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from stereodetect.inference.qwen_demo import (
    run_worker,
    score_completion,
    select_indices,
    task_messages,
    write_json,
)


class QwenDemoContracts(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "Sentence": f"Example {i}.",
                "Category": "gender",
                "Target": "School_Boys",
                "labels": str(i % 5),
            }
            for i in range(20)
        ]

    def test_sampling_and_input_isolation(self):
        indices = select_indices(self.rows, 5, 42, "balanced")
        self.assertEqual({int(self.rows[i]["labels"]) for i in indices}, set(range(5)))
        self.assertEqual(indices, select_indices(self.rows, 5, 42, "balanced"))
        self.assertEqual(len(set(select_indices(self.rows, 20, 42, "balanced"))), 20)
        for task in ("A", "B"):
            minimal = {"Sentence": self.rows[0]["Sentence"]}
            self.assertEqual(task_messages(task, self.rows[0]), task_messages(task, minimal))
        self.assertIn("Target: School_Boys", task_messages("C", self.rows[0])[1]["content"])
        with self.assertRaises(ValueError):
            select_indices(self.rows, 21, 42, "balanced")

    def test_json_fields_remain_independent_and_strict(self):
        text = '{"category":" GENDER ","target":"school_boys","label":0}'
        result = score_completion("B", text, self.rows[0], "stop")
        self.assertEqual(result["fields"], {"category": True, "target": False, "label": True})
        self.assertFalse(result["all_fields_correct"])
        for text in ('{"label":true}', '{"label":"0"}', "[]", "```json\n{}\n```"):
            result = score_completion("B", text, self.rows[0], "length")
            self.assertEqual(result["label"], -1)
            self.assertFalse(result["valid_schema"])
            self.assertTrue(result["truncated"])
        self.assertEqual(score_completion("A", "0", self.rows[0], "length")["label"], 0)
        self.assertEqual(score_completion("A", "00", self.rows[0], "length")["label"], -1)

    def test_worker_routes_three_adapters_and_disables_thinking(self):
        calls = []

        class Tokenizer:
            def encode(self, text, **kwargs):
                return [100 + int(text)] if text in {"0", "1", "2", "3", "4"} else [1, 2, 3]

            def apply_chat_template(self, messages, **kwargs):
                self_test.assertFalse(kwargs["enable_thinking"])
                self_test.assertFalse(kwargs["tokenize"])
                return "prompt text"

        class Engine:
            def __init__(self, **kwargs):
                pass

            def get_tokenizer(self):
                return Tokenizer()

            def generate(self, prompts, params, **kwargs):
                calls.append((prompts, params, kwargs))
                task = kwargs["lora_request"].lora_name[-1]
                return [
                    types.SimpleNamespace(
                        outputs=[
                            types.SimpleNamespace(
                                text=json.dumps(
                                    {"category": "gender", "target": "School_Boys", "label": 0}
                                )
                                if task == "B"
                                else "0",
                                finish_reason="stop",
                            )
                        ]
                    )
                    for _ in prompts
                ]

        def request(name, adapter_id, path):
            return types.SimpleNamespace(lora_name=name, lora_int_id=adapter_id, path=path)

        self_test = self
        modules = {
            "vllm": types.SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw),
            "vllm.lora": types.ModuleType("vllm.lora"),
            "vllm.lora.request": types.SimpleNamespace(LoRARequest=request),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "test.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            import hashlib

            plan = dict(
                csv=str(csv_path),
                data_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                indices=[0, 6, 12, 18, 4],
                output_dir=str(root),
                engine={"max_model_len": 2048},
                json_max_tokens=96,
                seed=42,
                mode="sft",
                adapters={k: k for k in "ABC"},
            )
            write_json(root / "config.json", plan)
            with patch.dict(sys.modules, modules):
                run_worker(root / "config.json")
            result = json.loads((root / "results.json").read_text())
            self.assertEqual([r["index"] for r in result["samples"]], plan["indices"])
            self.assertTrue(all(set(r["tasks"]) == set("ABC") for r in result["samples"]))
            self.assertEqual([c[2]["lora_request"].lora_int_id for c in calls], [1, 2, 3])
            self.assertEqual([c[1]["max_tokens"] for c in calls], [1, 96, 1])
            self.assertTrue(all(isinstance(prompt, str) for c in calls for prompt in c[0]))
            self.assertEqual(calls[0][1]["allowed_token_ids"], list(range(100, 105)))
            self.assertNotIn("allowed_token_ids", calls[1][1])


if __name__ == "__main__":
    unittest.main()

"""CPU regression coverage for task contracts and package boundaries."""

import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stereodetect.cli import COMMANDS, main
from stereodetect.data import load_json_rows, load_label_rows, read_rows
from stereodetect.metrics import detailed_report, metrics
from stereodetect.models.encoders.data import Examples
from stereodetect.models.qwen.data import JsonExamples, LabelExamples
from stereodetect.models.qwen.json_task import answer, messages, parse_label
from stereodetect.models.qwen.labels import messages as label_messages
from stereodetect.models.qwen.labels import sentence_only_for_adapter
from stereodetect.runtime import configure_runtime, package_versions


class TaskContracts(unittest.TestCase):
    def setUp(self):
        self.row = {
            "Category": "  GeNDer  ",
            "Target": "School_Boys",
            "Sentence": "Schoolboys were playing.",
            "labels": "3",
        }

    def test_json_input_and_answer(self):
        self.assertEqual(messages(self.row)[1]["content"], "Sentence: Schoolboys were playing.")
        self.assertEqual(
            answer(self.row), {"category": "gender", "target": "School_Boys", "label": 3}
        )
        self.assertIn("Target: School_Boys", label_messages(self.row)[1]["content"])

    def test_sentence_label_input_excludes_metadata(self):
        minimal = {"Sentence": self.row["Sentence"]}
        expected = label_messages(minimal, sentence_only=True)
        self.assertEqual(expected[1]["content"], "Sentence: Schoolboys were playing.")
        self.assertEqual(label_messages(self.row, sentence_only=True), expected)
        self.assertNotIn("provided category", expected[0]["content"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            self.assertFalse(sentence_only_for_adapter(adapter))
            (root / "run_config.json").write_text(json.dumps({"sentence_only": True}))
            self.assertTrue(sentence_only_for_adapter(adapter))
            (adapter / "run_config.json").write_text(json.dumps({"sentence_only": False}))
            self.assertFalse(sentence_only_for_adapter(adapter))

    def test_invalid_labels_count_as_errors(self):
        for value in ("{}", "[]", "null", '{"label":true}', '{"label":"1"}', '{"label":5}', "text"):
            with self.subTest(value=value):
                self.assertEqual(parse_label(value), -1)
        self.assertEqual(parse_label('{"label":0}'), 0)
        report = detailed_report([0, 1, 2, 3, 4], [0, -1, 2, 3, 4])
        self.assertEqual(report["invalid_predictions"], 1)
        self.assertEqual(report["confusion_matrix"][1][5], 1)
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), 5)
        self.assertEqual(report["accuracy"], 0.8)
        self.assertEqual(metrics([0], [0])["macro_f1"], 0.2)

    def test_csv_contracts_and_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.row))
                writer.writeheader()
                writer.writerow(self.row)
            self.assertEqual(load_json_rows(path), [self.row])
            self.assertEqual(load_label_rows(path)[0], [self.row])
            self.assertEqual(read_rows(path), [self.row])
            path.write_text("Sentence,labels\nhello,7\n")
            with self.assertRaises(ValueError):
                read_rows(path)
            with self.assertRaises(ValueError):
                load_json_rows(path)

    def test_dataset_masks_and_length_guards(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[1, 2] for _ in texts]}

            def apply_chat_template(self, *args, **kwargs):
                return [10, 11]

            def encode(self, text, **kwargs):
                return [12, 13]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.row))
                writer.writeheader()
                writer.writerow(self.row)
            tokenizer = Tokenizer()
            for cls in (LabelExamples, JsonExamples):
                with self.subTest(dataset=cls.__name__), contextlib.redirect_stdout(io.StringIO()):
                    data = cls(path, tokenizer, 4)
                    self.assertEqual(data[0]["labels"], [-100, -100, 12, 13])
                    with self.assertRaises(ValueError):
                        cls(path, tokenizer, 3)
            self.assertEqual(Examples(path, tokenizer, 2)[0]["labels"], 3)


class PackageBoundaries(unittest.TestCase):
    def test_all_command_help_without_optional_libraries(self):
        for group, commands in COMMANDS.items():
            for command in commands:
                with (
                    self.subTest(group=group, command=command),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as result:
                        main([group, command, "--help"])
                    self.assertEqual(result.exception.code, 0)

    def test_imports_do_not_load_gpu_libraries_or_change_environment(self):
        script = """
import importlib, os, sys
from stereodetect.cli import COMMANDS
before = dict(os.environ)
for commands in COMMANDS.values():
    for module, _ in commands.values():
        importlib.import_module('stereodetect.' + module)
assert dict(os.environ) == before
assert not {'torch', 'transformers', 'peft', 'vllm', 'wandb', 'shap'} & sys.modules.keys()
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_runtime_preserves_explicit_settings(self):
        with patch.dict(
            os.environ,
            {"TILELANG_CACHE_DIR": "/explicit", "STEREODETECT_CACHE_DIR": "/new"},
            clear=True,
        ):
            configure_runtime()
            self.assertEqual(os.environ["TILELANG_CACHE_DIR"], "/explicit")
        self.assertEqual(
            package_versions("stereodetect-missing-package"), {"stereodetect-missing-package": None}
        )

    def test_server_dry_run_uses_user_paths_and_does_not_write(self):
        from stereodetect.serving.qwen import main as serve

        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            output = Path(directory) / "output"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                serve(
                    [
                        "--adapter",
                        str(adapter),
                        "--output-dir",
                        str(output),
                        "--device",
                        "2",
                        "--model",
                        "/custom/model",
                        "--dry-run",
                    ]
                )
            record = json.loads(stream.getvalue())
            self.assertIn("/custom/model", record["command"])
            self.assertEqual(record["environment"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

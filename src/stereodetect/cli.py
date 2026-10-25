"""Command dispatch without importing model or GPU libraries."""

import argparse
from importlib import import_module

from stereodetect import __version__

# Each leaf delegates argument parsing to the corresponding reusable main(argv).
COMMANDS = {
    "train": {
        "albert": ("training.albert", "training"),
        "modernbert": ("training.modernbert", "training"),
        "qwen-json": ("training.qwen_json", "qwen"),
        "qwen-labels": ("training.qwen_labels", "qwen"),
    },
    "evaluate": {
        "qwen-json": ("evaluation.qwen_json", "qwen"),
        "qwen-labels": ("evaluation.qwen_labels", "qwen"),
        "qwen-thinking": ("evaluation.qwen_thinking", "qwen"),
        "qwen-vllm": ("evaluation.qwen_vllm", "client"),
        "json-fields": ("evaluation.json_fields", None),
    },
    "predict": {"qwen": ("inference.qwen", "qwen")},
    "explain": {
        "albert": ("explanations.encoder", "explanations"),
        "modernbert": ("explanations.modernbert", "explanations"),
        "samples": ("explanations.samples", "explanations"),
    },
    "report": {
        "loss-curves": ("reporting.loss_curves", "reporting"),
        "pdf": ("reporting.pdf", None),
    },
    "serve": {"qwen": ("serving.qwen", "serving")},
    "track": {"wandb": ("tracking.wandb_stream", "tracking")},
}


def main(argv=None):
    """Dispatch a command and return its exit status."""
    parser = argparse.ArgumentParser(prog="stereodetect", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    groups = parser.add_subparsers(dest="group", required=True)
    for group, commands in COMMANDS.items():
        group_parser = groups.add_parser(group)
        leaves = group_parser.add_subparsers(dest="command", required=True)
        for command, (module, extra) in commands.items():
            leaf = leaves.add_parser(command, add_help=False)
            leaf.set_defaults(module=module, extra=extra)
    args, remaining = parser.parse_known_args(argv)
    try:
        result = import_module(f"stereodetect.{args.module}").main(remaining)
    except ModuleNotFoundError as error:
        if error.name and not error.name.startswith("stereodetect"):
            install = f"stereodetect[{args.extra}]" if args.extra else "stereodetect"
            parser.error(
                f"Missing dependency {error.name!r}. Install with: pip install '{install}'"
            )
        raise
    return result if isinstance(result, int) else 0

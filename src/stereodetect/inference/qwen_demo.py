"""Colorful, no-thinking vLLM demo of all three Qwen3.5 StereoDetect tasks."""

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stereodetect.data import load_json_rows, normalize_category
from stereodetect.labels import LABEL_NAMES
from stereodetect.models.qwen import json_task, labels

REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ADAPTERS = {
    "A": "outputs/kaggle_qwen35_sentence_lora_1022-183119/results/artifacts/adapter",
    "B": "outputs/qwen35_2b_stereodetect_json/adapter",
    "C": "outputs/qwen35_2b_stereodetect/adapter",
}
TASKS = {
    "A": "Sentence → label",
    "B": "Sentence → category, target, label (JSON)",
    "C": "Sentence + gold category + gold target → label",
}


def select_indices(rows, count, seed, sampling):
    """Choose before inference; balanced sampling cycles through shuffled label buckets."""
    if not 1 <= count <= len(rows):
        raise ValueError(f"Sample count must be between 1 and {len(rows)}")
    rng = random.Random(seed)
    if sampling == "random":
        return rng.sample(range(len(rows)), count)
    buckets = [[i for i, row in enumerate(rows) if int(row["labels"]) == k] for k in range(5)]
    for bucket in buckets:
        rng.shuffle(bucket)
    selected = []
    while len(selected) < count:
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
    return selected


def task_messages(task, row):
    if task == "B":
        return json_task.messages(row)
    return labels.messages(row, sentence_only=task == "A")


def score_completion(task, text, row, finish_reason):
    """Preserve strict parsing and independent JSON field checks used in evaluation."""
    result = {"completion": text, "finish_reason": finish_reason}
    if task != "B":
        value = text.strip()
        prediction = int(value) if value in {"0", "1", "2", "3", "4"} else -1
        return dict(result, label=prediction, label_correct=prediction == int(row["labels"]))
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    obj = obj if isinstance(obj, dict) else {}
    category = obj.get("category")
    target = obj.get("target")
    label = json_task.parse_label(text)
    checks = {
        "category": isinstance(category, str)
        and normalize_category(category) == normalize_category(row["Category"]),
        "target": isinstance(target, str) and target == row["Target"],
        "label": label == int(row["labels"]),
    }
    return dict(
        result,
        label=label,
        label_correct=checks["label"],
        fields=checks,
        all_fields_correct=all(checks.values()),
        valid_schema=set(obj) == {"category", "target", "label"}
        and isinstance(category, str)
        and normalize_category(category) in json_task.CATEGORIES
        and isinstance(target, str)
        and bool(target.strip())
        and label != -1,
        truncated=finish_reason == "length",
    )


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_worker(plan_path):
    """Keep engine output in a log and release CUDA resources when this process exits."""
    from stereodetect.runtime import configure_runtime

    configure_runtime()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    plan = json.loads(Path(plan_path).read_text())
    source = Path(plan["csv"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan["data_sha256"]:
        raise ValueError("Dataset changed after selecting demo samples")
    rows = load_json_rows(source)
    selected = [rows[i] for i in plan["indices"]]
    output = Path(plan["output_dir"])
    started = time.perf_counter()
    llm = LLM(**plan["engine"])
    tokenizer = llm.get_tokenizer()
    digits = [tokenizer.encode(str(k), add_special_tokens=False) for k in range(5)]
    if any(len(ids) != 1 for ids in digits):
        raise ValueError("Each classification label must encode to exactly one token")
    digit_ids = [ids[0] for ids in digits]
    records = [
        {"index": i, "gold": json_task.answer(row), "sentence": row["Sentence"], "tasks": {}}
        for i, row in zip(plan["indices"], selected)
    ]
    timings = {}
    for adapter_id, task in enumerate(TASKS, 1):
        write_json(output / "progress.json", {"stage": f"Running Task {task}: {TASKS[task]}"})
        prompts = [
            tokenizer.apply_chat_template(
                task_messages(task, row),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in selected
        ]
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Tokenizer must return text prompts for vLLM")
        token_budget = plan["json_max_tokens"] if task == "B" else 1
        if any(
            len(tokenizer.encode(prompt, add_special_tokens=False)) + token_budget
            > plan["engine"]["max_model_len"]
            for prompt in prompts
        ):
            raise ValueError("A sample exceeds --max-model-len; increase that option")
        params = SamplingParams(
            temperature=0,
            max_tokens=token_budget,
            seed=plan["seed"],
            **({"allowed_token_ids": digit_ids} if task != "B" else {}),
        )
        request = None
        if plan["mode"] == "sft":
            request = LoRARequest(f"stereodetect-{task}", adapter_id, plan["adapters"][task])
        task_started = time.perf_counter()
        predictions = llm.generate(
            prompts,
            params,
            lora_request=request,
            use_tqdm=False,
        )
        timings[task] = time.perf_counter() - task_started
        if len(predictions) != len(selected):
            raise RuntimeError("vLLM did not return one prediction per sample")
        for record, row, prediction in zip(records, selected, predictions):
            completion = prediction.outputs[0]
            record["tasks"][task] = score_completion(
                task,
                completion.text,
                row,
                completion.finish_reason,
            )
    write_json(
        output / "results.json",
        {
            "config": plan,
            "samples": records,
            "task_seconds": timings,
            "total_seconds": time.perf_counter() - started,
        },
    )


def render_results(console, result):
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    def label(value):
        return f"{value} · {LABEL_NAMES[value]}" if value in range(5) else "Invalid prediction"

    def verdict(correct):
        return Text(
            "✓ Match" if correct else "✗ Mismatch", style="bold green" if correct else "bold red"
        )

    for number, record in enumerate(result["samples"], 1):
        gold = record["gold"]
        console.print()
        console.rule(
            f"[bold cyan]Sample {number}/{len(result['samples'])} · test CSV row {record['index'] + 2}"
        )
        console.print(
            Panel(
                Text(record["sentence"], style="bold white"), title="Sentence", border_style="cyan"
            )
        )
        console.print(
            Text(
                f"Ground truth  {label(gold['label'])}  |  Category: {gold['category']}  |  Target: {gold['target']}",
                style="dim",
            )
        )
        table = Table(box=box.ROUNDED, expand=True, show_lines=True)
        table.add_column("Task / model input", style="cyan", ratio=3)
        table.add_column("Predicted label", ratio=3)
        table.add_column("Check", width=12)
        for task, title in TASKS.items():
            prediction = record["tasks"][task]
            table.add_row(
                Text(f"{task} · {title}"),
                Text(label(prediction["label"])),
                verdict(prediction["label_correct"]),
            )
        console.print(table)
        prediction = record["tasks"]["B"]
        try:
            displayed_json = json.dumps(
                json.loads(prediction["completion"]), indent=2, ensure_ascii=False
            )
        except (ValueError, TypeError):
            displayed_json = prediction["completion"]
        field_checks = Text()
        for field, correct in prediction["fields"].items():
            field_checks.append(
                f"{'✓' if correct else '✗'} {field}   ", style="green" if correct else "red"
            )
        if not prediction["valid_schema"] or prediction["truncated"]:
            field_checks.append("Invalid JSON/schema or truncated output", style="bold red")
        console.print(
            Panel(
                Group(
                    Syntax(displayed_json, "json", theme="monokai", word_wrap=True),
                    field_checks,
                ),
                title="Task B · actual generated JSON",
                border_style="magenta",
            )
        )
        console.print(
            Text(
                f"Task C received these gold annotations: category={gold['category']}, target={gold['target']}",
                style="dim",
            )
        )
    console.print()
    summary = Table(title="Demo results · selected samples only", box=box.ROUNDED, expand=True)
    summary.add_column("Task", ratio=4)
    summary.add_column("Label matches", justify="right")
    summary.add_column("Batch inference", justify="right")
    count = len(result["samples"])
    for task, title in TASKS.items():
        correct = sum(row["tasks"][task]["label_correct"] for row in result["samples"])
        summary.add_row(
            Text(f"{task} · {title}"), f"{correct}/{count}", f"{result['task_seconds'][task]:.2f}s"
        )
    console.print(summary)
    joint = sum(row["tasks"]["B"]["all_fields_correct"] for row in result["samples"])
    console.print(
        Text(
            f"Task B: all three fields match in {joint}/{count} samples. These are demo counts, not full-test metrics.",
            style="dim",
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/test.csv")
    parser.add_argument("--model", default=labels.MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument(
        "--mode",
        choices=("sft", "base"),
        default="sft",
        help="Default: three task-specific trained LoRA adapters",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument(
        "--sampling",
        choices=("balanced", "random"),
        default="balanced",
        help="Balanced selects one sample per label for the default five samples",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", help="CUDA_VISIBLE_DEVICES value, such as a GPU index or MIG UUID"
    )
    for task in TASKS:
        parser.add_argument(f"--adapter-{task.lower()}", default=ADAPTERS[task])
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--json-max-tokens", type=int, default=96)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0,
        help="Optional vLLM CPU weight offload; slower, but uses less GPU memory",
    )
    parser.add_argument("--kv-cache-mib", type=int, help="Optional fixed GPU KV cache in MiB")
    parser.add_argument(
        "--output-dir", help="Fresh output directory for results, settings and vLLM logs"
    )
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--_worker-plan", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker_plan:
        run_worker(args._worker_plan)
        return 0
    if (
        not 0 < args.gpu_memory_utilization <= 1
        or args.cpu_offload_gb < 0
        or args.json_max_tokens < 1
        or args.max_model_len <= args.json_max_tokens
        or (args.kv_cache_mib is not None and args.kv_cache_mib < 1)
    ):
        parser.error("Invalid memory fraction, offload size or token budget")
    try:
        source = Path(args.csv).expanduser().resolve()
        rows = load_json_rows(source)
        indices = select_indices(rows, args.num_samples, args.seed, args.sampling)
        adapters = {
            task: str(Path(getattr(args, f"adapter_{task.lower()}")).expanduser().resolve())
            for task in TASKS
        }
        ranks = []
        if args.mode == "sft":
            for task, adapter in adapters.items():
                config = json.loads((Path(adapter) / "adapter_config.json").read_text())
                if not (Path(adapter) / "adapter_model.safetensors").is_file():
                    raise ValueError(f"Task {task}: missing adapter weights at {adapter}")
                ranks.append(config["r"])
            if not labels.sentence_only_for_adapter(adapters["A"]):
                raise ValueError("Task A requires an adapter trained with sentence_only=true")
            if labels.sentence_only_for_adapter(adapters["C"]):
                raise ValueError("Task C requires an annotation-conditioned adapter")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console(
        force_terminal={"auto": None, "always": True, "never": False}[args.color],
        no_color=args.color == "never",
        highlight=False,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir or f"outputs/demo/qwen35_{stamp}").expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new/empty --output-dir to preserve previous demo results")
    output.mkdir(parents=True, exist_ok=True)
    engine = dict(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_seqs=min(5, args.num_samples),
        max_num_batched_tokens=512,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=True,
        language_model_only=True,
        generation_config="vllm",
        gdn_prefill_backend="triton",
        enable_prefix_caching=False,
        enable_lora=args.mode == "sft",
    )
    if args.mode == "sft":
        engine.update(max_loras=1, max_cpu_loras=3, max_lora_rank=max(ranks))
    if args.kv_cache_mib is not None:
        engine["kv_cache_memory_bytes"] = args.kv_cache_mib * 1024 * 1024
    plan = dict(
        model=args.model,
        mode=args.mode,
        engine=engine,
        adapters=adapters if args.mode == "sft" else {},
        csv=str(source),
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        indices=indices,
        sampling=args.sampling,
        seed=args.seed,
        enable_thinking=False,
        json_max_tokens=args.json_max_tokens,
        output_dir=str(output),
        device=args.device or os.environ.get("CUDA_VISIBLE_DEVICES", "environment default"),
        input_columns={
            "A": ["Sentence"],
            "B": ["Sentence"],
            "C": ["Sentence", "Category", "Target"],
        },
    )
    write_json(output / "config.json", plan)
    console.print(
        Panel(
            Text(
                f"StereoDetect  /  Qwen3.5-2B\nvLLM · {args.mode.upper()} · thinking OFF · greedy decoding\n"
                f"{len(indices)} test samples · {args.sampling} selection · seed {args.seed}",
                style="bold cyan",
            ),
            border_style="cyan",
            title="Three-task live inference demo",
        )
    )
    for task, title in TASKS.items():
        console.print(Text(f"{task}  {title}"))
    console.print(
        Text("Green = matches ground truth   Red = differs from ground truth", style="dim")
    )
    console.print(Text(f"Engine log: {output / 'vllm.log'}", style="dim"))
    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.device
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # vLLM 0.22 can try to warm up a missing DeepGEMM even for BF16 models.
    # This demo uses BF16 throughout and does not need its FP8 kernels.
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MAX_JOBS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        env.setdefault(key, "2" if key != "TORCHINDUCTOR_COMPILE_THREADS" else "1")
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    )
    with (output / "vllm.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stereodetect.inference.qwen_demo",
                "--_worker-plan",
                str(output / "config.json"),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with console.status(
                "Loading Qwen3.5 in vLLM (first launch can take a few minutes)…", spinner="dots"
            ) as status:
                while process.poll() is None:
                    progress = output / "progress.json"
                    if progress.exists():
                        status.update(json.loads(progress.read_text())["stage"])
                    time.sleep(0.2)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            console.print("[yellow]Demo interrupted.[/yellow]")
            return 130
    if process.returncode:
        console.print(
            Text(
                f"vLLM inference failed (exit {process.returncode}). See {output / 'vllm.log'}",
                style="bold red",
            )
        )
        console.print(
            Text(
                "Check available GPU memory, --device, and your CUDA/compiler environment.",
                style="dim",
            )
        )
        return 1
    result = json.loads((output / "results.json").read_text())
    render_results(console, result)
    console.print(Text(f"\nSaved predictions: {output / 'results.json'}", style="bold cyan"))
    transcript = Console(file=io.StringIO(), width=100, record=True, highlight=False)
    render_results(transcript, result)
    transcript.save_text(str(output / "terminal.txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

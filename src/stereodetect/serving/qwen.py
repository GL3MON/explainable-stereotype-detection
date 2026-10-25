"""Launch a vLLM JSON adapter server with bounded CPU and compiler resources."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from stereodetect.models.qwen.labels import MODEL
from stereodetect.runtime import configure_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/eval/json_test_vllm_thinking_limited_20261014"
    )
    parser.add_argument("--model", default=MODEL, help="Cached base model ID or local weights path")
    parser.add_argument("--adapter", default="outputs/qwen35_2b_stereodetect_json/adapter")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with vLLM installed")
    parser.add_argument(
        "--device", help="CUDA_VISIBLE_DEVICES for the server; defaults to the environment"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--cpu-count", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--compiler-jobs", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without creating files or launching a server",
    )
    args = parser.parse_args(argv)
    if min(args.max_num_seqs, args.cpu_count, args.threads, args.compiler_jobs) < 1:
        parser.error("Sequence, CPU, thread, and compiler counts must be positive")
    if not 0 < args.gpu_memory_utilization <= 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid memory fraction or port")
    adapter = Path(args.adapter).expanduser().resolve()
    if not (adapter / "adapter_config.json").is_file():
        parser.error(f"No adapter configuration at {adapter}")
    interpreter = shutil.which(args.python)
    if interpreter is None:
        parser.error(f"Python executable not found: {args.python}")
    output = Path(args.output_dir).expanduser().resolve()
    pid_path = output / "server.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text())
        status = Path(f"/proc/{pid}/status")
        if status.exists() and "\nState:\tZ" not in status.read_text():
            parser.error(f"Saved server PID {pid} still exists; refusing duplicate launch")
    configure_runtime()
    cpus = (
        sorted(os.sched_getaffinity(0))[: args.cpu_count]
        if hasattr(os, "sched_getaffinity")
        else None
    )
    cmd = [
        interpreter,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        "qwen35-base",
        "--tokenizer",
        str(adapter),
        "--enable-lora",
        "--lora-modules",
        f"stereodetect-json={adapter}",
        "--max-lora-rank",
        "16",
        "--reasoning-parser",
        "qwen3",
        "--reasoning-config",
        '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        "1024",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--language-model-only",
        "--generation-config",
        "vllm",
        "--gdn-prefill-backend",
        "triton",
    ]
    limits = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MAX_JOBS": str(args.compiler_jobs),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "PYTHONFAULTHANDLER": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        limits[key] = str(args.threads)
    if args.device is not None:
        limits["CUDA_VISIBLE_DEVICES"] = args.device
    env = os.environ.copy()
    env.update(limits)
    # Preserve the original launcher's avoidance of conflicting loader overrides.
    env.pop("LD_LIBRARY_PATH", None)
    record = {"command": cmd, "environment": limits, "cpu_affinity": cpus, "nice": 10}
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "LIBRARY_PATH", "CC", "CXX", "NVCC_CCBIN"):
        if key in env:
            record["environment"][key] = env[key]
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return

    def bounded():
        if cpus is not None:
            os.sched_setaffinity(0, cpus)
        if hasattr(os, "nice"):
            os.nice(10)

    output.mkdir(parents=True, exist_ok=True)
    (output / "server_command.json").write_text(json.dumps(record, indent=2) + "\n")
    with (output / "server.log").open("a") as log:
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=bounded if os.name == "posix" else None,
        )
    pid_path.write_text(f"{process.pid}\n")
    print(f"Started PID {process.pid}; log {output / 'server.log'}")


if __name__ == "__main__":
    main()

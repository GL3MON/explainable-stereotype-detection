"""Kaggle entry point: set up dependencies, run two ranks, and archive results."""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def status(state, **extra):
    payload = {"state": state, "time": time.time(), **extra}
    (WORK / "job_status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("JOB_STATUS", json.dumps(payload), flush=True)


def run(command, env=None):
    print("COMMAND", json.dumps(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    try:
        status("setup")
        run(["nvidia-smi"])
        archives = list(Path("/kaggle/input").rglob("stereodetect_training_bundle.zip"))
        workspace = WORK / "workspace"
        workspace.mkdir(exist_ok=True)
        if len(archives) == 1:
            with zipfile.ZipFile(archives[0]) as z:
                z.extractall(workspace)
        else:
            # Kaggle commonly expands dataset ZIP uploads before mounting inputs.
            bundles = [p.parent for p in Path("/kaggle/input").rglob("manifest.json")
                       if (p.parent / "train.py").is_file() and (p.parent / "data/train.csv").is_file()]
            assert len(bundles) == 1, f"Expected one expanded training bundle, found {bundles}"
            shutil.copytree(bundles[0], workspace, dirs_exist_ok=True)
        # Reuse the Kaggle CUDA/PyTorch stack; isolate the Hugging Face dependencies.
        venv = Path("/tmp/stereodetect_med")
        run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)])
        python = str(venv / "bin/python")
        run([sys.executable, "-m", "pip", "--python", python, "install", "--disable-pip-version-check",
             "transformers==5.17.0", "peft==0.20.0", "accelerate>=1.1,<2", "scikit-learn>=1.4"])
        env = os.environ.copy()
        env.update(PYTHONPATH=str(workspace / "src"), TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
                   PYTORCH_ALLOC_CONF="expandable_segments:True", NCCL_DEBUG="WARN")
        cache = Path("/tmp/stereodetect_models")
        env["HF_HOME"] = str(cache)
        status("download_model")
        code = (
            "from huggingface_hub import snapshot_download; "
            "p=snapshot_download('Qwen/Qwen3.5-2B',"
            "revision='15852e8c16360a2fea060d615a32b45270f8a8fc',"
            "allow_patterns=['*.json','*.safetensors','*.jinja','*.txt']); "
            "print('MODEL_PATH='+p)"
        )
        process = subprocess.run([python, "-c", code], env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(process.stdout, flush=True)
        process.check_returncode()
        model_path = next(line.removeprefix("MODEL_PATH=") for line in process.stdout.splitlines()
                          if line.startswith("MODEL_PATH="))
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        shutil.copy(workspace / "manifest.json", OUT / "input_manifest.json")
        shutil.copy(workspace / "train.py", OUT / "train.py")
        status("training", world_size=2)
        command = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(workspace / "train.py"), "--data-dir", str(workspace / "data"),
                   "--output-dir", str(OUT), "--model-path", model_path, "--epochs", "3"]
        with (WORK / "training.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"Distributed training exited with code {returncode}; see training.log")
        assert (OUT / "completion.json").is_file(), "Missing final evaluation"
        status("complete", results=json.loads((OUT / "completion.json").read_text()))
    except Exception as error:
        status("failed", error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise
    finally:
        # Publish a compact downloadable bundle, including diagnostic files on failure.
        with zipfile.ZipFile(WORK / "training_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for path in OUT.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(WORK))
            for name in ("job_status.json", "training.log"):
                path = WORK / name
                if path.exists():
                    z.write(path, name)


if __name__ == "__main__":
    main()

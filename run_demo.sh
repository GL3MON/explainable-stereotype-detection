#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

demo_device="${DEMO_CUDA_DEVICE:-}"
if [[ -z "$demo_device" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "$(nvidia-smi -L)" == *"MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b"* ]]; then
    demo_device="MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b"
  fi
fi

demo_args=(--cpu-offload-gb 3 --kv-cache-mib 128 --gpu-memory-utilization 0.14)
if [[ -n "$demo_device" ]]; then
  demo_args+=(--device "$demo_device")
fi
exec conda run --no-capture-output -n med python demo_qwen.py "${demo_args[@]}" "$@"

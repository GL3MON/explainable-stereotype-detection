# Installation

StereoDetect requires Python 3.11 or newer. Run commands from a working directory containing
your `data/` directory, or pass explicit dataset and output paths.

```bash
python -m pip install -e .
stereodetect --help
python -m stereodetect --version
```

Choose extras for the workflows you need:

| Extra | Provides |
| --- | --- |
| `training` | ALBERT and ModernBERT training, including W&B logging |
| `qwen` | Qwen LoRA training, local evaluation, and prediction |
| `explanations` | Encoder scoring, LIME, SHAP, and explanation figures |
| `reporting` | Saved loss curve plots |
| `client` | HTTP evaluation against an existing vLLM endpoint |
| `serving` | Local vLLM server |
| `tracking` | W&B logging for evaluation and explanations, and log streaming |
| `dev` | Build and lint tooling |

```bash
python -m pip install -e '.[training,qwen,explanations,reporting,client,tracking,dev]'
```

Install a PyTorch build suitable for your CUDA runtime before installing GPU extras.
vLLM also constrains the compatible PyTorch/CUDA stack; use a separate serving environment
when its requirements differ from your training environment. The launcher accepts
`--python /path/to/serving/python`.

The version ranges in `pyproject.toml` are dependency requirements, not a reproduced
environment lock. Recorded experiment environments remain in their original W&B logs.
Historical runs recorded Transformers 5.17.0, PEFT 0.20.0, SHAP 0.53.0,
PyTorch 2.11.0+cu129, and vLLM 0.22.0+cu129.

For an existing configured environment, install just this package without resolving dependencies:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Qwen local workflows load cached weights with `local_files_only=True`.
ALBERT and ModernBERT follow Transformers cache/download behavior; set
`HF_HUB_OFFLINE=1` to require cached weights. PDF rendering requires Chrome or Chromium
on `PATH`.

## Runtime configuration

Select a GPU through `CUDA_VISIBLE_DEVICES` or the server's `--device` option.
The package respects `CUDA_HOME`, `CC`, `CXX`, `NVCC_CCBIN`, `LIBRARY_PATH`, and
`TILELANG_CACHE_DIR`. Optional convenience settings are:

```bash
export STEREODETECT_CUDA_HOME=/path/to/cuda-or-conda-environment
export STEREODETECT_CACHE_DIR=/path/to/tilelang-cache
```

The CUDA setting discovers Conda compiler executables only if they exist and preserves
explicit compiler variables. Configure `LIBRARY_PATH` yourself if your toolchain needs
CUDA linker stubs. These settings are applied when running a Qwen workflow, never on import.

## Development

```bash
python -m unittest discover -s tests -v
python -m build
ruff check src tests
```

The CPU tests cover task input contracts, metrics, datasets, CLI help, imports, and server
command construction. GPU model execution needs a compatible environment and cached models.

# StereoDetect

Training, evaluation, inference, and explanations for five-class stereotype detection
with ALBERT, ModernBERT, and Qwen3.5.

## Install

Requires Python 3.11 or newer. From the project root:

```bash
python -m pip install -e .
stereodetect --help
```

Install workflow dependencies as needed:

```bash
python -m pip install -e '.[training,qwen,explanations,reporting,client,tracking]'
```

For an existing configured environment, use
`python -m pip install --no-deps --no-build-isolation -e .`.
See [installation](docs/installation.md) for GPU setup, optional extras, offline models,
and runtime configuration.

## Use

```bash
stereodetect train albert --data-dir data
stereodetect train modernbert --data-dir data
stereodetect train qwen-json --data-dir data
stereodetect evaluate qwen-json --split test
stereodetect explain modernbert --help
stereodetect report loss-curves --help
```

`python -m stereodetect` exposes the same interface. Every workflow accepts `--help`.
Install the package to run its commands.

See [usage and command mapping](docs/usage.md) for all 17 commands and path conventions.
Relative paths resolve from the working directory. Data, saved models, and experiment
outputs remain external to the installed package.

## Tasks

| Workflow | Input | Output |
| --- | --- | --- |
| ALBERT / ModernBERT | Sentence | Five-class label |
| Qwen JSON | Sentence | Category, target, and label JSON |
| Qwen label-only (task A) | Sentence | Numeric label |
| Qwen annotation-conditioned (task C) | Sentence, gold category, gold target | Numeric label |

Labels: **0** Anti-stereotype, **1** Stereotype, **2** Neutral without target term,
**3** Neutral with target term, **4** Bias. Dataset details are in
[data/README.md](data/README.md).

JSON training/evaluation scores the label field. Invalid or missing labels count as
incorrect predictions and appear in an extra confusion-matrix column. Category and
target accuracy can be measured separately with `stereodetect evaluate json-fields`.

## Project report

- [Full project report (PDF)](report/StereoDetect_Project_Report.pdf)
- [Editable LaTeX source and figures](report/StereoDetect_LaTeX_Report.zip)
- [LaTeX build instructions](report/latex/README.md)

The report compares Base and task-specific SFT evaluations for all three Qwen
tasks, including confusion matrices showing the effect of auxiliary category
and target annotations. Raw dataset CSVs, model weights, run outputs, credentials,
and local visual QA files are excluded from this repository.

## Project structure

```text
src/stereodetect/     Installable package and CLI
  models/            Encoder scoring, datasets, Qwen prompts and trainer
  training/          ALBERT, ModernBERT, and Qwen training
  evaluation/        Local/API evaluation and JSON field validation
  inference/         Adapter prediction
  explanations/      LIME/SHAP analysis and sample reports
  reporting/         Loss plots and PDF rendering
  serving/           Configurable vLLM launcher
  tracking/          W&B log streaming
docs/                Installation, usage, and experiment records
tests/               CPU regression checks
data/                Local dataset CSVs and label definitions
outputs/             Local checkpoints, predictions, and reports
wandb/               Local experiment logs
```

## Experiment records

- [Qwen workflows](docs/experiments/QWEN_WORKFLOWS.md)
- [ALBERT reproduction](docs/experiments/ALBERT_REPRODUCTION.md)
- [ModernBERT reproduction](docs/experiments/MODERNBERT_REPRODUCTION.md)
- [ModernBERT requested settings](docs/experiments/MODERNBERT_REQUESTED_SETTINGS.md)
- [ModernBERT large preset](docs/experiments/MODERNBERT_LARGE_PRESET.md)

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check src tests
python -m build
```

Distributions contain the package code and omit datasets, model weights, and run logs.

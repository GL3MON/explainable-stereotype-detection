# Usage

`stereodetect` and `python -m stereodetect` expose the same commands. Every command accepts
`--help`, and each workflow also exposes a Python `main(argv=None)` function.

| Command | Purpose |
| --- | --- |
| `stereodetect train albert` | Train the ALBERT sentence classifier |
| `stereodetect train modernbert` | Train the ModernBERT sentence classifier |
| `stereodetect train qwen-json` | Train sentence-to-JSON generation |
| `stereodetect train qwen-labels` | Train category/target-conditioned label prediction |
| `stereodetect evaluate qwen-json` | Evaluate generated JSON labels locally |
| `stereodetect evaluate qwen-labels` | Evaluate label-token scores locally |
| `stereodetect evaluate qwen-thinking` | Evaluate local thinking completions |
| `stereodetect evaluate qwen-vllm` | Evaluate thinking completions through the API |
| `stereodetect evaluate json-fields` | Validate saved category and target predictions |
| `stereodetect predict qwen` | Classify a sentence using the label adapter |
| `stereodetect explain albert` | Generate encoder LIME/SHAP explanations |
| `stereodetect explain modernbert` | Generate explanations with ModernBERT defaults |
| `stereodetect explain samples` | Build the curated explanation sample report |
| `stereodetect report loss-curves` | Plot saved training and validation losses |
| `stereodetect report pdf` | Render a saved explanation report as PDF |
| `stereodetect serve qwen` | Launch the vLLM adapter server |
| `stereodetect track wandb` | Stream an existing training log to W&B |

Install the package before running commands. Python integrations should import from
`stereodetect`.

## Train and evaluate

```bash
stereodetect train modernbert --data-dir data --output-dir outputs/my-modernbert
stereodetect train qwen-json --data-dir data --output-dir outputs/my-qwen-json
stereodetect evaluate qwen-json --adapter outputs/my-qwen-json/adapter \
  --split test --output-dir outputs/eval/my-json-test
stereodetect predict qwen 'Schoolboys were playing.' --category Gender --target Schoolboys
```

ALBERT and ModernBERT classify only `Sentence`. The JSON Qwen task sees only `Sentence`
and generates `category`, `target`, and `label`. The label-only Qwen task sees `Category`,
`Target`, and `Sentence`. Preserve that distinction when comparing experiments.

The dataset schema and labels are described in [data/README.md](../data/README.md).
Labels are integers 0–4. JSON label scoring counts invalid predictions as incorrect (`-1`)
and includes an invalid column in the confusion matrix. Category/target field validation
is available through `evaluate json-fields`.

## Explain and report

```bash
stereodetect explain albert --model-dir outputs/albert_stereodetect/model
stereodetect report loss-curves --run-dir outputs/my-modernbert
stereodetect report pdf outputs/albert_stereodetect/explanations/index.html
```

ModernBERT explanation defaults retain the original paired ALBERT sample configuration.
Pass `--indices-json`, `--predictions`, `--model-dir`, and `--output-dir` to select other
artifacts. The curated sample report retains its original experiment-specific indices
and commentary; use it with that experiment's matching artifacts.

## Serve and benchmark

```bash
stereodetect serve qwen --device 0 --adapter outputs/my-qwen-json/adapter --dry-run
stereodetect serve qwen --device 0 --adapter outputs/my-qwen-json/adapter
stereodetect evaluate qwen-vllm --adapter outputs/my-qwen-json/adapter \
  --output-dir outputs/eval/my-thinking-test
```

The server runs in the background, recording its command, PID, and log in the output
directory. It uses cached base weights, defaults to localhost port 8011, and inherits
the selected device from the environment unless `--device` is supplied. `--dry-run`
prints the resolved command without writing files or starting a server.

## Paths and artifacts

Relative paths resolve from the current working directory, including after installation.
Pass absolute paths when running elsewhere. Datasets, checkpoints, reports, W&B logs,
and model caches are external artifacts and are excluded from package distributions.
Existing output directory defaults and overwrite/resume checks are retained.

> Experiment notes retained from the original workspace. Commands use the installed CLI
> and run from the project root. Machine-specific GPU and environment details describe
> the recorded runs; see [installation](../installation.md) for portable setup.

# Sentence-to-JSON training

`stereodetect train qwen-json` now learns to generate `{"category": "gender", "target": "Sisters", "label": 0}` from the **sentence alone**. Category, target, and gold label are never provided as input. Categories are normalized by trimming, case folding, and collapsing whitespace: `Gender` and `gender` both become `gender`. Targets retain their original strings, including capitalization and underscores. Labels remain integers 0–4 with the definitions in [data/README.md](../../data/README.md).

All JSON fields contribute to supervised training loss. Evaluation generates JSON with thinking disabled and scores **only the label**: accuracy, macro precision/recall/F1, weighted F1, and per-class scores. Incorrect or missing category/target fields do not change the label score. Invalid JSON or a missing/non-integer/out-of-range label counts as an incorrect label (`-1`), never as a dropped example. The confusion matrix has an extra invalid-prediction column. The best checkpoint is selected by validation label macro F1.

The existing `med` environment, cached Qwen3.5-2B base model, LoRA settings, and W&B logging are retained. The new model writes to a separate `outputs/qwen35_2b_stereodetect_json` directory. Previous label-only adapters and results are preserved. No new training or GPU evaluation was launched while preparing this change.

Train:

```bash
CUDA_VISIBLE_DEVICES=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  stereodetect train qwen-json
```

Training saves validation completions at each evaluation, the best `adapter/`, `test_completions.jsonl`, `test_predictions.csv`, `evaluation.json`, and `classification_report.txt`. `--eval-batch-size` defaults to 16, and `--max-new-tokens` defaults to 96. JSON validation generates completions, so it takes longer than the earlier one-token scoring. Future resume checkpoints must belong to this JSON task.

Standalone evaluation after JSON training:

```bash
CUDA_VISIBLE_DEVICES=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  stereodetect evaluate qwen-json --split test --wandb
```

It saves complete JSON responses, numeric label predictions, label-only metrics, and a text classification report under `outputs/eval/json_test`. It refuses to overwrite existing completions; choose a new `--output-dir` for a rerun.

The older label-only workflow is retained below; its training entry point is now `stereodetect train qwen-labels`. The existing `stereodetect evaluate qwen-labels`, `stereodetect predict qwen`, and `stereodetect evaluate qwen-thinking` continue to use that workflow and the original adapter.

---

# StereoDetect with Qwen3.5-2B

Offline LoRA fine-tuning of the cached `Qwen/Qwen3.5-2B` model using the existing `/home/harishr/miniconda3/envs/med` environment. No new dependencies are installed.

The input contains `Category`, `Target`, and `Sentence`. The output is one label: **0** Anti-stereotype, **1** Stereotype, **2** Neutral without target term, **3** Neutral with target term, **4** Bias. Both training and prediction explicitly use `enable_thinking=False`. Only the assistant's numeric answer and end token contribute to training loss.

Training uses `data/train.csv` (6,080 rows), selects the best checkpoint by macro F1 on `data/val.csv` (868 rows), and then evaluates once on `data/test.csv` (1,738 rows). Predictions choose the highest-scoring token among labels 0–4, so they cannot include explanations. Reported probabilities are normalized across these five labels, rather than calibrated confidence estimates.

Defaults: three epochs, LoRA rank 16/alpha 32/dropout 0.05 on language-model attention and feed-forward projections, BF16, learning rate 0.0002, cosine schedule with 5% warmup, effective batch size 16, gradient checkpointing, seed 42. The scripts point the existing TileLang/CUDA backend to the C++ compiler already installed in the med environment. No compiler or libraries are installed.

W&B logging uses the existing configured account and the `stereodetect` project. The current run streams all training and validation metrics, including the history before logging was attached, through `stereodetect track wandb`; later training runs use the native Transformers W&B integration. Final test metrics, per-class scores, and the confusion matrix are also logged by the streamer. The run link is stored in `outputs/qwen35_2b_stereodetect/wandb_run.json`. Use `--wandb-project` and `--wandb-run-name` to customize future runs.

Train:

```bash
CUDA_VISIBLE_DEVICES=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  stereodetect train qwen-labels
```

The GPU UUID is the least occupied H100 partition observed at setup time; check availability before later runs. Training requires permission to write checkpoints and use the GPU.

Outputs in `outputs/qwen35_2b_stereodetect/`: `adapter/`, `run_config.json`, `trainer_state.json`, `evaluation.json`, `test_results.json`, and `test_predictions.csv`. The adapter requires the original cached base model. Resume an interrupted run with `--resume outputs/qwen35_2b_stereodetect/checkpoint-N`.

Predict after training:

```bash
CUDA_VISIBLE_DEVICES=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  stereodetect predict qwen \
  'Schoolboys were playing.' --category Gender --target Schoolboys
```

Model source: https://huggingface.co/Qwen/Qwen3.5-2B
Dataset label definitions: [data/README.md](../../data/README.md).

Evaluate a saved adapter after training finishes:

```bash
CUDA_VISIBLE_DEVICES=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  stereodetect evaluate qwen-labels \
  --adapter outputs/qwen35_2b_stereodetect/adapter \
  --split test --batch-size 4 --wandb
```

`--adapter` also accepts a `checkpoint-N` directory. Use `--split val` for validation, `--csv PATH` for a custom CSV with the same columns, or `--base-only` instead of `--adapter` for a base-model baseline. `--limit N` makes a partial evaluation, which is explicitly marked in the report. No model evaluation has been run while training is active.

Evaluation uses the training prompt with thinking disabled. It feeds only prompts into the model, uses left padding for batched scoring, and selects the highest-scoring token among labels 0–4. It loads cached weights and existing libraries without installing or downloading anything. Missing adapters produce a clear error.

Default evaluation outputs are `outputs/eval/adapter_test/evaluation.json` and `predictions.csv`; override with `--output-dir PATH`. The report contains accuracy, macro F1, weighted F1, per-class precision/recall/F1, category scores, a confusion matrix (rows=true labels, columns=predictions), and log loss computed from probabilities normalized across the five label tokens. These probabilities are not calibrated confidence estimates. Macro F1 always includes all five labels, including absent labels as zero, for consistent comparisons across subsets.

Add `--wandb` to create a separate evaluation run in the `stereodetect` project, logging aggregate metrics, per-class/category summaries, and the confusion matrix. It saves the run URL to the evaluation output directory. Evaluation predictions and dataset sentences are saved locally rather than uploaded by the evaluator. Training remains a separate W&B run.


ALBERT and HEARTS-style explanations: see [ALBERT_REPRODUCTION.md](ALBERT_REPRODUCTION.md) for training, MIG 2 commands, method details, and saved artifacts. The ALBERT experiment is a sentence-only five-label classifier.


## vLLM thinking benchmark on MIG1

`stereodetect serve qwen` serves the newer sentence-to-JSON LoRA adapter from `outputs/qwen35_2b_stereodetect_json/adapter` with its saved tokenizer. The base weights remain cached locally. The API listens at `http://127.0.0.1:8011/v1`, and requests select model `stereodetect-json`. Thinking is explicitly enabled through `chat_template_kwargs`.

The launcher uses `med`, MIG device 1 (`MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1`), four-core CPU affinity, low process priority, two compiler jobs, two library threads, Triton GDN prefill, native PyTorch sampling, and eager execution. GPU memory utilization is capped at 60%. These limits are separate from Docker's 24 GiB RAM limit; monitor `/sys/fs/cgroup/memory.current` and `memory.events`, rather than host figures from `free`.

Start serving:

```bash
stereodetect serve qwen
```

Evaluate the complete test split after the API is ready:

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  taskset -c 0-3 nice -n 10 stereodetect evaluate qwen-vllm --concurrency 32 --max-tokens 2048 \
  --output-dir outputs/eval/json_test_vllm_thinking_limited_20261014
```

Only the sentence enters the model. The final JSON label is scored; missing or invalid labels count as incorrect (`-1`). All responses, reasoning, final answers, finish reasons, and token counts are saved. Defaults follow Qwen's thinking text sampling settings: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, and seed 42. The 2,048-token cap includes reasoning and final JSON; truncated responses are reported separately. Qwen documents thinking loops for this 2B model, so this benchmark measures thinking with this explicit token cap.

The evaluator writes `completions.jsonl`, `predictions.csv`, `evaluation.json`, and `classification_report.txt`. An identical command resumes saved completions; differing configurations require a new output directory. Previous non-thinking results remain in their original output directories.

`med` saves `MAX_JOBS=2`, `CUDA_HOME`, and `LIBRARY_PATH` through conda environment variables. The linker path includes `targets/x86_64-linux/lib/stubs` and `targets/x86_64-linux/lib`, resolving `-lcuda` while the real system NVIDIA driver is used at runtime. Reactivate `med` to apply these settings in an existing terminal. The launcher applies them directly and removes `LD_LIBRARY_PATH` overrides that caused API import crashes.


ModernBERT comparison: see [MODERNBERT_REPRODUCTION.md](MODERNBERT_REPRODUCTION.md) for the matched sentence-only classifier, paired LIME/SHAP explanations, and PDF commands.


Requested ModernBERT retraining settings and updated defaults: [MODERNBERT_REQUESTED_SETTINGS.md](MODERNBERT_REQUESTED_SETTINGS.md).


ModernBERT-large regularized experiment: [MODERNBERT_LARGE_PRESET.md](MODERNBERT_LARGE_PRESET.md).


## Sentence-only numeric-label LoRA

Use `stereodetect train qwen-labels --sentence-only` to train from `Sentence` alone and
predict one digit 0–4 with thinking disabled. Category and target remain available
locally for reports but never enter the prompt. The default output directory is
`outputs/qwen35_2b_stereodetect_sentence_labels`, preserving the other adapters.
The best adapter is selected by validation macro F1 and then evaluated on test.

For a memory-limited GPU, use `--batch-size 1 --gradient-accumulation 16` to retain
effective batch size 16. `--report-to none` saves metrics locally without W&B.
Saved adapters include the input contract in `run_config.json`, which
`stereodetect evaluate qwen-labels --adapter PATH` and
`stereodetect predict qwen "SENTENCE" --adapter PATH` read automatically.
For a base-model sentence-only baseline, add `--base-only --sentence-only` to evaluation.

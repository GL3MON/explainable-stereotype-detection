> Experiment notes retained from the original workspace. Commands use the installed CLI
> and run from the project root. Machine-specific GPU and environment details describe
> the recorded runs; see [installation](../installation.md) for portable setup.

> The script defaults were subsequently changed to the user's requested three-epoch preset. See [MODERNBERT_REQUESTED_SETTINGS.md](MODERNBERT_REQUESTED_SETTINGS.md) for the current defaults and flags to reproduce this original baseline. Original artifacts remain in the directory documented below.

# ModernBERT StereoDetect baseline and explanations

`stereodetect train modernbert` fine-tunes `answerdotai/ModernBERT-base` to classify sentences into the same five labels as the ALBERT baseline. Category/target metadata are not included as input. ModernBERT is an encoder classifier: it returns class probabilities and does not generate JSON or reasoning.

Training uses the same existing CSV splits, six epochs, effective batch size 64 (32 x 2 accumulation), learning rate 2e-5, linear decay, no warmup, no weight decay, maximum length 128, seed 42, and BF16 autocast as the ALBERT baseline. No training examples are truncated. The best checkpoint is selected by validation macro F1. ModernBERT uses its supported SDPA attention configuration; PyTorch, vLLM, and Transformers are not patched or upgraded. Pretrained weights are downloaded, with no package installation needed. The experiment runs in conda `med` on MIG device 2.

```bash
CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b \
  stereodetect train modernbert
```

Saved model and classifier reports: `outputs/modernbert_stereodetect/model/`, `classification_report.txt`, `evaluation.json`, and `test_predictions.csv`. The ALBERT comparison is in `comparison_with_albert.json` and `comparison_predictions.csv`. This is a single matched run per model, not a multi-seed comparison.

## Paired explanations

```bash
CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b \
  stereodetect explain modernbert --wandb
```

The wrapper reuses the shared LIME/Kernel SHAP implementation and the **same 100 selected test indices** as ALBERT. The original sample was balanced by gold label and enriched with ALBERT errors. It remains a diagnostic sample; its correctness proportion and explanation summaries are not full-test statistics.

Explanations use aligned word occurrences (`bow=False`), each model's mask token, LIME with 1,000 perturbations, and Kernel SHAP with up to 512 coalitions and an all-masked background. Both predicted-class probability and fixed stereotype-class probability are explained. SHAP additivity is checked. Word scores, figures, masking-probability-drop checks, and raw model probabilities are saved under `outputs/modernbert_stereodetect/explanations/`.

**Interpretation:** Word contributions are allocations under this masking game, not independent word probabilities or proof of causal language understanding. Masking may disrupt phrases. LIME is a local additive approximation; SHAP allocates interaction effects across word features. Agreement between methods is not calibrated prediction correctness confidence. The ALBERT and ModernBERT predicted-class explanations may concern different classes when the models disagree; the fixed stereotype-class explanations concern the same class.

Render all expanded examples to a PDF using the already installed Chrome browser:

```bash
stereodetect report pdf \
  outputs/modernbert_stereodetect/explanations/index.html
```

Outputs include `index.html`, `explanations.pdf`, `explanations.jsonl`, `figures/*.png`, `summary.json`, `summary.csv`, and W&B links. Existing explanation output is not overwritten: choose another `--output-dir` for a rerun. To select a different diagnostic sample, pass `--indices-json ""`; to explain all examples, also pass `--num-examples 0`.

References: [ModernBERT model](https://huggingface.co/answerdotai/ModernBERT-base), [HEARTS method](https://aclanthology.org/2025.ijcnlp-long.1.pdf), [ALBERT reproduction notes](ALBERT_REPRODUCTION.md).

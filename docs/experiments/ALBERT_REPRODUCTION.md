> Experiment notes retained from the original workspace. Commands use the installed CLI
> and run from the project root. Machine-specific GPU and environment details describe
> the recorded runs; see [installation](../installation.md) for portable setup.

# ALBERT training and HEARTS-style explanations

This experiment uses `albert/albert-base-v2` with a five-class classification head on StereoDetect. The input is `Sentence` only; labels are the existing integers 0–4. Category and Target are retained as metadata. The trained classifier does not generate JSON or reasoning.

Training ran in the existing `med` environment on MIG device 2 (`MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b`). It used the HEARTS appendix settings: six epochs, learning rate 2e-5, effective batch size 64 (32 per batch, two gradient accumulation steps). This adaptation uses NVIDIA/BF16, a maximum sequence length of 128, seed 42, the existing train/val/test splits, and five labels rather than HEARTS' binary EMGSD task. No sentence was truncated during training. The checkpoint is selected by validation macro F1, then evaluated on all 1,738 test examples.

The existing PyTorch, vLLM, and Transformers installations were not changed. Only the user-authorized SHAP, LIME, and required slicer packages were added with `--no-deps`. Protected package versions and distribution RECORD hashes were checked against `outputs/albert_environment_before.json`.

Training artifacts are in `outputs/albert_stereodetect/`: the selected full model in `model/`, epoch checkpoints, `evaluation.json`, `classification_report.txt`, `test_predictions.csv`, and the run configuration and trainer state.

## Training command

```bash
CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b \
  stereodetect train albert
```

A completed model is already saved. Choose a new `--output-dir` for another run. Resume an unfinished run with `--resume outputs/albert_stereodetect/checkpoint-N`.

## Explanation command

```bash
CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b \
  stereodetect explain albert \
  --num-examples 100 --lime-samples 1000 --shap-samples 512 --wandb
```

LIME and Kernel SHAP explain the **same word occurrences**, preserving punctuation and word order. Both replace absent words with ALBERT's mask token. Repeated words are separate features (`bow=False`). LIME uses its default cosine-distance kernel, all word features, and 1,000 perturbation samples. Kernel SHAP uses an all-masked background, identity probability link, up to 512 sampled coalitions, and no L1 feature selection. Short sentences may enumerate all nontrivial coalitions rather than sample them.

Each example explains both its predicted class and stereotype class 1. This preserves the stereotype-probability question in HEARTS while also explaining the five-way classifier decision. SHAP values are checked to sum to the predicted probability minus the background probability. A synthetic linear example verified recovery of known word contributions before model explanations were run.

The default diagnostic sample contains 20 examples per gold label, with up to half misclassified examples per label. Its agreement summaries are **not representative test-set statistics**. Set `--num-examples 0` to explain the entire test set. Choose another `--output-dir` for a rerun; saved explanations are not overwritten.

Outputs in `outputs/albert_stereodetect/explanations/`:

- `explanations.jsonl`: raw word-level LIME/SHAP scores, probabilities, gold/predicted labels, agreement, and masking results.
- `figures/`: standalone PNG comparison figures, displaying the 20 most influential word positions where necessary.
- `index.html`: local highlighted-word report and figure links.
- `summary.json` and `summary.csv`: cosine/Pearson agreement, summaries for correct/incorrect predictions, and changes in probability after masking the highest-ranked 20% of words versus a random 20%.
- `run_config.json`: sample indices, package versions, feature definitions, masking, and reproducibility settings.

Red highlights support the explained class; blue highlights oppose it. Undefined agreement for constant/zero attribution vectors is recorded as null rather than forced to zero. Agreement between methods is not calibrated correctness confidence. Word masking can create unnatural text; probability-drop checks are diagnostic, not proof of causal linguistic explanations. StereoDetect has no gold importance annotations, so HEARTS' gold stereotype-token hit rate cannot be reproduced from the target-group annotations.

References: [HEARTS paper, Sections 3.3 and A.2](https://aclanthology.org/2025.ijcnlp-long.1.pdf), [repository](https://github.com/holistic-ai/HEARTS-Text-Stereotype-Detection), [LIME](https://lime-ml.readthedocs.io/en/latest/lime.html), [Kernel SHAP](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html).

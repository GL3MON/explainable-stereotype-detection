> Experiment notes retained from the original workspace. Commands use the installed CLI
> and run from the project root. Machine-specific GPU and environment details describe
> the recorded runs; see [installation](../installation.md) for portable setup.

# ModernBERT with the requested training settings

The new model remains `answerdotai/ModernBERT-base`, trained on sentence-only StereoDetect inputs and the same five labels and CSV splits. No PyTorch, vLLM, or Transformers code/packages were changed, and bitsandbytes was not installed. Standard AdamW (`adamw_torch`) is used as requested.

Training: batch size 32; gradient accumulation 1; warmup steps 5; epochs 3; learning rate 5e-5; weight decay 0.001; cosine learning-rate schedule; seed and data seed 3407; logging every step; W&B reporting. Precision uses BF16 when supported, otherwise FP16. On MIG 2, BF16 is supported and was used.

There are 570 optimizer steps. Validation and checkpoint saving run every 10% of total steps, meaning steps 57, 114, 171, 228, 285, 342, 399, 456, 513, 570. The best checkpoint is selected by validation label macro F1. Test evaluation uses all 1,738 examples.

The new artifacts are in `outputs/modernbert_stereodetect_lr5e5_cosine_ep3_seed3407/`. The first ModernBERT experiment and ALBERT experiment are preserved.

```bash
CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b \
  stereodetect train modernbert
```

These requested settings are now the script defaults. For another fresh run, supply a new `--output-dir`; the script refuses to overwrite an existing completed model. To reproduce the original six-epoch baseline, explicitly pass `--epochs 6 --gradient-accumulation 2 --learning-rate 2e-5 --optim adamw_torch --warmup-steps 0 --weight-decay 0 --lr-scheduler-type linear --seed 42 --logging-steps 10 --eval-strategy epoch`, plus a new output directory.

Saved classifier artifacts: `model/`, checkpoints, `classification_report.txt`, `evaluation.json`, `test_predictions.csv`, trainer state, and run configuration. `comparison_with_previous_runs.json` compares the first ModernBERT run, this requested preset, and ALBERT. Several hyperparameters and the seed changed, so the improvement cannot be attributed solely to learning rate or optimizer.

Paired explanations use the same 100 diagnostic test examples, LIME/SHAP settings, and perturbation seed as ALBERT and the first ModernBERT run. They explain word-level sensitivity under the same masking scheme, not independent word probabilities. They are stored in this run's `explanations/` directory along with raw word scores, PNG figures, HTML, and an expanded PDF. The sample is enriched with ALBERT errors and is not representative test statistics.

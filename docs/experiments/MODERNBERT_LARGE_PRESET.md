> Experiment notes retained from the original workspace. Commands use the installed CLI
> and run from the project root. Machine-specific GPU and environment details describe
> the recorded runs; see [installation](../installation.md) for portable setup.

# ModernBERT-large regularized preset

Model: `answerdotai/ModernBERT-large`. Inputs remain sentence-only; output is a five-label classifier. The existing train, validation, and test CSVs are used unchanged. Candidate annotation conflicts remain in the separate audit for review.

Training uses conda `med` and MIG device 2. There are three epochs; effective batch size 32 (micro-batch 16, accumulation 2); learning rate 3e-5; 30 warmup optimizer steps; cosine decay; weight decay 0.01; standard AdamW; seed/data seed 3407; BF16 autocast; sequence limit 128; and logging every step. No packages or framework code are changed. All original training examples fit the sequence limit without truncation.

There are 570 optimizer steps. Validation and checkpoint saving run every 57 steps. The selected model maximizes validation macro F1 and is evaluated on the test set only after selection.

```bash
CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b \
  stereodetect train modernbert \
  --model answerdotai/ModernBERT-large \
  --output-dir outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407 \
  --epochs 3 --batch-size 16 --gradient-accumulation 2 \
  --learning-rate 3e-5 --warmup-steps 30 --weight-decay 0.01 \
  --optim adamw_torch --lr-scheduler-type cosine --seed 3407 \
  --run-name modernbert-large-lr3e5-warmup30-wd001-epoch3
```

The output directory contains `model/`, the retained checkpoints, `run_config.json`, `trainer_state.json`, `evaluation.json`, `classification_report.txt`, and `test_predictions.csv`. Use another output directory for a fresh rerun, or `--resume` for an interrupted checkpoint. Earlier base-model artifacts are preserved.

Plot loss after training:

```bash
stereodetect report loss-curves \
  --run-dir outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407
```

The existing explanation runner also supports this model: override `--model-dir`, `--predictions`, and `--output-dir` to the large-model artifacts and choose `--display-name ModernBERT-large`. It reuses ALBERT's 100 diagnostic indices for paired comparisons. Its word scores remain sensitivity measurements under word masking, not independent word probabilities. The 0.95 macro-F1 objective is an experimental target, not an assumed result.

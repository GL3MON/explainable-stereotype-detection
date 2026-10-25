"""Plot saved Trainer training/validation loss without rerunning the model."""

import argparse
import csv
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", default="outputs/modernbert_stereodetect_lr5e5_cosine_ep3_seed3407"
    )
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if args.window < 1:
        parser.error("--window must be positive")
    run = Path(args.run_dir)
    state = json.loads((run / "trainer_state.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    model_name = config.get("model", "Model").split("/")[-1]
    training = [r for r in state["log_history"] if "loss" in r]
    validation = [r for r in state["log_history"] if "eval_loss" in r]
    if not training or not validation:
        parser.error("Saved state must contain both training and validation losses")
    steps = np.array([r["step"] for r in training])
    losses = np.array([r["loss"] for r in training])
    ends = np.arange(1, len(losses) + 1)
    starts = np.maximum(ends - args.window, 0)
    sums = np.r_[0, np.cumsum(losses)]
    smoothed = (sums[ends] - sums[starts]) / (ends - starts)
    eval_steps = np.array([r["step"] for r in validation])
    eval_losses = np.array([r["eval_loss"] for r in validation])
    best_path = state.get("best_model_checkpoint")
    best_step = int(Path(best_path).name.split("-")[-1]) if best_path else None
    minimum = int(eval_losses.argmin())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot(
        steps, losses, color="#2974a6", alpha=0.2, linewidth=0.75, label="Training loss: each batch"
    )
    ax.plot(
        steps,
        smoothed,
        color="#175e91",
        linewidth=2.3,
        label=f"Training loss: {args.window}-step trailing mean",
    )
    ax.plot(
        eval_steps,
        eval_losses,
        color="#d66328",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Validation loss: full split",
    )
    if best_step is not None:
        ax.axvline(
            best_step,
            color="#65726c",
            linestyle="--",
            linewidth=1.2,
            label=f"Saved checkpoint: best validation macro F1 (step {best_step})",
        )
    ax.annotate(
        f"Lowest validation loss: {eval_losses[minimum]:.4f}\nStep {eval_steps[minimum]}",
        xy=(eval_steps[minimum], eval_losses[minimum]),
        xytext=(-120, 55),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#d66328"},
        fontsize=10,
        color="#9a471f",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#e4d4ca", "alpha": 0.95},
    )
    ax.set(
        title=f"{model_name}: training and validation loss",
        xlabel="Optimizer step",
        ylabel="Cross-entropy loss",
    )
    ax.set_xlim(0, steps[-1] + 8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    caption = (
        f"{config['epochs']:g} epochs | effective batch {config['effective_batch_size']} | {config.get('optim', 'AdamW')} | "
        f"LR {config['learning_rate']:g}, {config.get('lr_scheduler_type', 'linear')} decay | seed {config['seed']}\n"
        f"{len(validation)} validation checks. The saved checkpoint is selected by macro F1, rather than minimum loss."
    )
    fig.text(0.12, 0.02, caption, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(run / f"loss_curves.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (run / "loss_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "step", "epoch", "loss", "training_trailing_mean"]
        )
        writer.writeheader()
        for row, average in zip(training, smoothed):
            writer.writerow(
                {
                    "split": "train",
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "loss": row["loss"],
                    "training_trailing_mean": float(average),
                }
            )
        for row in validation:
            writer.writerow(
                {
                    "split": "validation",
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "loss": row["eval_loss"],
                    "training_trailing_mean": "",
                }
            )
    print(f"Plots: {run / 'loss_curves.png'}, {run / 'loss_curves.pdf'}")
    print(f"Data: {run / 'loss_history.csv'}")
    print(f"Lowest validation loss: {eval_losses[minimum]:.6f} at step {eval_steps[minimum]}")


if __name__ == "__main__":
    main()

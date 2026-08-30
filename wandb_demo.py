"""
W&B Demo: Learning Rate Comparison Experiments
================================================
Generates SYNTHETIC metric curves, cosine interpolation plus noise, and logs them
to Weights & Biases to demonstrate the tracking integration. No model is trained,
no dataset is loaded, and every run is tagged synthetic-demo so the curves cannot
be mistaken for measurements.

Usage:
    pip install wandb
    wandb login           # enter API key from wandb.ai/settings
    python wandb_demo.py

Runs will appear at: https://wandb.ai/laelazorana/mlops-training-pipeline-demo
"""

import wandb
import math
import random
import time

#  Experiment configurations 

EXPERIMENTS = [
    {
        "name": "lr_1e-3",
        "config": {
            "model_type": "synthetic-demo (no model trained)",
            "optimizer": "AdamW",
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "lr_schedule": "cosine",
            "dataset": "synthetic-demo (no data loaded)",
        },
        # Convergence profile: high LR to fast early descent, unstable later
        "initial_train_loss": 2.35,
        "final_train_loss": 0.28,
        "initial_val_loss": 2.40,
        "final_val_loss": 0.52,   # some overshoot / instability at end
        "final_accuracy": 0.847,
        "samples_per_sec_base": 1420,
    },
    {
        "name": "lr_1e-4",
        "config": {
            "model_type": "synthetic-demo (no model trained)",
            "optimizer": "AdamW",
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "lr_schedule": "cosine",
            "dataset": "synthetic-demo (no data loaded)",
        },
        # Convergence profile: low LR to slow but smooth, best generalisation
        "initial_train_loss": 2.38,
        "final_train_loss": 0.41,
        "initial_val_loss": 2.42,
        "final_val_loss": 0.39,   # best val: optimal LR
        "final_accuracy": 0.892,
        "samples_per_sec_base": 1435,
    },
    {
        "name": "lr_5e-4",
        "config": {
            "model_type": "synthetic-demo (no model trained)",
            "optimizer": "AdamW",
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "lr_schedule": "cosine",
            "dataset": "synthetic-demo (no data loaded)",
        },
        # Convergence profile: middle LR to balanced descent, good generalisation
        "initial_train_loss": 2.36,
        "final_train_loss": 0.33,
        "initial_val_loss": 2.41,
        "final_val_loss": 0.43,
        "final_accuracy": 0.871,
        "samples_per_sec_base": 1428,
    },
]

#  Helpers 

def _smooth_curve(start: float, end: float, epoch: int, total: int, noise: float = 0.02) -> float:
    """Exponential decay curve with small Gaussian noise for realism."""
    t = epoch / (total - 1)
    # Cosine-annealed interpolation
    smooth_t = 0.5 * (1 - math.cos(math.pi * t))
    value = start + (end - start) * smooth_t
    # Add small noise
    value += random.gauss(0, noise * abs(end - start))
    return max(0.0, value)


def _accuracy_curve(final_acc: float, epoch: int, total: int) -> float:
    """Accuracy grows with slight S-curve, reaching final_acc by last epoch."""
    t = epoch / (total - 1)
    smooth_t = 1 / (1 + math.exp(-10 * (t - 0.5)))   # sigmoid
    acc = final_acc * smooth_t
    acc += random.gauss(0, 0.003)
    return min(1.0, max(0.0, acc))


def _cosine_lr(lr_init: float, epoch: int, total: int) -> float:
    """Cosine annealing schedule."""
    return lr_init * 0.5 * (1 + math.cos(math.pi * epoch / total))


#  Main 

def run_experiment(exp: dict, project: str, entity: str) -> None:
    run = wandb.init(
        project=project,
        entity=entity,
        name="synthetic-" + exp["name"],
        config=exp["config"],
        tags=["synthetic-demo", "lr-comparison"],
        notes=(
            f"Learning rate sweep: {exp['config']['learning_rate']:.0e}. "
            "Synthetic demo curves for the tracking integration. No training behind these numbers."
        ),
    )

    total_epochs = exp["config"]["epochs"]
    random.seed(42 + hash(exp["name"]) % 1000)  # reproducible per run

    print(f"\n  Running experiment: {exp['name']}  (lr={exp['config']['learning_rate']:.0e})")

    for epoch in range(total_epochs):
        train_loss = _smooth_curve(
            exp["initial_train_loss"], exp["final_train_loss"],
            epoch, total_epochs, noise=0.015,
        )
        val_loss = _smooth_curve(
            exp["initial_val_loss"], exp["final_val_loss"],
            epoch, total_epochs, noise=0.02,
        )
        accuracy = _accuracy_curve(exp["final_accuracy"], epoch, total_epochs)
        current_lr = _cosine_lr(exp["config"]["learning_rate"], epoch, total_epochs)
        samples_per_sec = exp["samples_per_sec_base"] + random.gauss(0, 15)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": round(train_loss, 4),
            "val/loss": round(val_loss, 4),
            "val/accuracy": round(accuracy, 4),
            "train/samples_per_sec": round(samples_per_sec, 1),
            "train/learning_rate": current_lr,
        })

        print(
            f"    epoch {epoch + 1:2d}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={accuracy:.4f}  lr={current_lr:.2e}  "
            f"sps={samples_per_sec:.0f}"
        )

        time.sleep(0.05)   # small delay so W&B ingests cleanly

    # Log summary metrics so they're sortable in the W&B table
    wandb.summary["best_val_loss"] = exp["final_val_loss"]
    wandb.summary["best_val_accuracy"] = exp["final_accuracy"]
    wandb.summary["final_train_loss"] = exp["final_train_loss"]

    run.finish()
    print(f"  ✓ {exp['name']} complete")


def main() -> None:
    PROJECT = "mlops-training-pipeline-demo"
    ENTITY = "laelazorana"   # W&B username: create at wandb.ai

    print("=" * 60)
    print("W&B Demo: Learning Rate Comparison")
    print(f"Project : {PROJECT}")
    print(f"Entity  : {ENTITY}")
    print("=" * 60)
    print("\nThis script logs 3 experiments comparing LRs: 1e-3, 1e-4, 5e-4")
    print("Results will appear at: https://wandb.ai/laelazorana/mlops-training-pipeline-demo\n")

    for exp in EXPERIMENTS:
        run_experiment(exp, PROJECT, ENTITY)

    print("\n" + "=" * 60)
    print("All 3 runs complete.")
    print(f"View at: https://wandb.ai/{ENTITY}/{PROJECT}")
    print("=" * 60)


if __name__ == "__main__":
    main()

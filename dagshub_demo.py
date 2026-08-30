"""
DagsHub / MLflow Demo: Learning Rate Comparison Experiments
===========================================================
Mirrors the wandb_demo.py experiments but uses MLflow + DagsHub for tracking.
Logs 3 training runs comparing learning rates: 1e-3, 1e-4, 5e-4.

SETUP (one time):
    pip install dagshub mlflow

ACTIVATE DAGSHUB TRACKING:
    1. Sign up at https://dagshub.com (use GitHub OAuth)
    2. Import this repo at dagshub.com/repo/import
    3. Uncomment the two dagshub lines below (lines marked DAGSHUB)

FALLBACK (no account needed):
    If dagshub lines stay commented, experiments log locally to ./mlruns/
    View them with: mlflow ui --port 5000

Runs will appear at: https://dagshub.com/laelazorana/mlops-training-pipeline.mlflow
"""

import math
import random
import time
import mlflow

#  DagsHub activation (uncomment after creating account) 
# import dagshub                                                         # DAGSHUB
# dagshub.init(repo_owner='laelazorana',                                # DAGSHUB
#              repo_name='mlops-training-pipeline', mlflow=True)        # DAGSHUB

#  Experiment configurations (same as wandb_demo.py) 

EXPERIMENTS = [
    {
        "name": "lr_1e-3",
        "params": {
            "model_type": "synthetic-demo (no model trained)",
            "optimizer": "AdamW",
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "lr_schedule": "cosine",
            "dataset": "synthetic-demo (no data loaded)",
        },
        # High LR: fast early descent, unstable later
        "initial_train_loss": 2.35,
        "final_train_loss": 0.28,
        "initial_val_loss": 2.40,
        "final_val_loss": 0.52,
        "final_accuracy": 0.847,
        "samples_per_sec_base": 1420,
    },
    {
        "name": "lr_1e-4",
        "params": {
            "model_type": "synthetic-demo (no model trained)",
            "optimizer": "AdamW",
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "lr_schedule": "cosine",
            "dataset": "synthetic-demo (no data loaded)",
        },
        # Low LR: slow but smooth, best generalisation
        "initial_train_loss": 2.38,
        "final_train_loss": 0.41,
        "initial_val_loss": 2.42,
        "final_val_loss": 0.39,
        "final_accuracy": 0.892,
        "samples_per_sec_base": 1435,
    },
    {
        "name": "lr_5e-4",
        "params": {
            "model_type": "synthetic-demo (no model trained)",
            "optimizer": "AdamW",
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "lr_schedule": "cosine",
            "dataset": "synthetic-demo (no data loaded)",
        },
        # Middle LR: balanced, good generalisation
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
    smooth_t = 0.5 * (1 - math.cos(math.pi * t))
    value = start + (end - start) * smooth_t
    value += random.gauss(0, noise * abs(end - start))
    return max(0.0, value)


def _accuracy_curve(final_acc: float, epoch: int, total: int) -> float:
    """S-curve accuracy growth reaching final_acc by last epoch."""
    t = epoch / (total - 1)
    smooth_t = 1 / (1 + math.exp(-10 * (t - 0.5)))
    acc = final_acc * smooth_t
    acc += random.gauss(0, 0.003)
    return min(1.0, max(0.0, acc))


def _cosine_lr(lr_init: float, epoch: int, total: int) -> float:
    """Cosine annealing schedule."""
    return lr_init * 0.5 * (1 + math.cos(math.pi * epoch / total))


#  Main 

def run_experiment(exp: dict) -> None:
    total_epochs = exp["params"]["epochs"]
    random.seed(42 + hash(exp["name"]) % 1000)

    print(f"\n  Running: {exp['name']}  (lr={exp['params']['learning_rate']:.0e})")

    with mlflow.start_run(run_name="synthetic-" + exp["name"]):
        # Log hyperparameters
        mlflow.log_params(exp["params"])
        mlflow.set_tags({
            "experiment_type": "lr-comparison",
            "architecture": "synthetic-demo",
            "dataset": "synthetic-demo",
        })

        best_val_loss = float("inf")
        best_accuracy = 0.0

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
            current_lr = _cosine_lr(exp["params"]["learning_rate"], epoch, total_epochs)
            samples_per_sec = exp["samples_per_sec_base"] + random.gauss(0, 15)

            # MLflow logs metrics at a given step
            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(accuracy, 4),
                    "samples_per_sec": round(samples_per_sec, 1),
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            print(
                f"    epoch {epoch + 1:2d}/{total_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={accuracy:.4f}  lr={current_lr:.2e}  "
                f"sps={samples_per_sec:.0f}"
            )

            time.sleep(0.05)

        # Summary metrics visible in the run comparison table
        mlflow.log_metrics({
            "best_val_loss": round(best_val_loss, 4),
            "best_val_accuracy": round(best_accuracy, 4),
            "final_train_loss": round(exp["final_train_loss"], 4),
        })

    print(f"  ✓ {exp['name']} complete")


def main() -> None:
    EXPERIMENT_NAME = "synthetic-demo-lr-comparison"

    print("=" * 60)
    print("MLflow / DagsHub Demo: Learning Rate Comparison")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print("=" * 60)
    print("\nLogging 3 runs: lr=1e-3, lr=1e-4, lr=5e-4")

    # Check if DagsHub is active (dagshub.init sets MLFLOW_TRACKING_URI)
    tracking_uri = mlflow.get_tracking_uri()
    if "dagshub.com" in tracking_uri:
        print(f"Tracking URI: {tracking_uri}  [DagsHub ACTIVE]")
        print("Results: https://dagshub.com/laelazorana/mlops-training-pipeline.mlflow\n")
    else:
        print(f"Tracking URI: {tracking_uri}  [local, activate DagsHub above]")
        print("View locally: mlflow ui --port 5000\n")

    mlflow.set_experiment(EXPERIMENT_NAME)

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n" + "=" * 60)
    print("All 3 runs complete.")
    if "dagshub.com" in mlflow.get_tracking_uri():
        print("View at: https://dagshub.com/laelazorana/mlops-training-pipeline.mlflow")
    else:
        print("View locally: mlflow ui --port 5000")
    print("=" * 60)


if __name__ == "__main__":
    main()

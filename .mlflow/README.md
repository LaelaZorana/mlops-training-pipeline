# MLflow Experiment Structure

This directory is created by MLflow when running experiments locally (fallback mode).

## Experiments

| Experiment | Description |
|---|---|
| `synthetic-demo-lr-comparison` | 3-run synthetic LR sweep, 1e-3, 1e-4, 5e-4, generated curves, no model trained |

## Tracked Metrics (per epoch)

| Metric | Description |
|---|---|
| `train_loss` | Training cross-entropy loss |
| `val_loss` | Validation loss |
| `val_accuracy` | Validation accuracy, 0 to 1 |
| `samples_per_sec` | Training throughput |
| `learning_rate` | Current LR (cosine annealed) |

## Summary Metrics (per run)

| Metric | Description |
|---|---|
| `best_val_loss` | Lowest val loss across all epochs |
| `best_val_accuracy` | Highest accuracy across all epochs |
| `final_train_loss` | Train loss at epoch 10 |

## Local View

```bash
mlflow ui --port 5000
# Open http://127.0.0.1:5000
```

## DagsHub (Remote Tracking)

When DagsHub is activated in `dagshub_demo.py`, runs are pushed to:

```
https://dagshub.com/laelazorana/mlops-training-pipeline.mlflow
```

The DagsHub MLflow server is provisioned automatically when you import the repo, no server setup required.

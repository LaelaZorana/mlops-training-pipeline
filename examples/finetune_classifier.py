"""
Example: Fine-tuning a HuggingFace classifier with ExperimentTracker.

Shows the full workflow:
  1. Load a pre-trained HuggingFace model
  2. Prepare a text classification dataset
  3. Wrap in our Trainer with experiment tracking
  4. Train for a few epochs
  5. Evaluate and log results

Requires: transformers, datasets, torch
Run: python examples/finetune_classifier.py
"""

import sys

# ── Dependency checks ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("torch not installed. pip install torch")
    sys.exit(0)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HF_AVAILABLE = True
except ImportError:
    print("transformers not installed. Using a dummy model for demonstration.")
    _HF_AVAILABLE = False

from pipeline.trainer import Trainer
from pipeline.experiment_tracker import ExperimentTracker


# ── Dummy model for when HuggingFace isn't available ──────────────────────────

class DummyClassifier(nn.Module):
    """Simple MLP classifier for demonstration without HuggingFace."""
    def __init__(self, input_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_dummy_dataloader(num_samples: int = 200, input_dim: int = 64, num_classes: int = 2):
    """Create a simple synthetic dataset for demonstration."""
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def finetune_with_hf(model_name: str = "distilbert-base-uncased"):
    """
    Fine-tune a HuggingFace model on a classification task.

    In production you'd load real labeled data here. We use synthetic
    embeddings for the demo since downloading the SST-2 dataset takes time.
    """
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Sample texts and labels (positive/negative sentiment)
    texts = [
        "This movie was absolutely fantastic!",
        "I hated every minute of it.",
        "Great performance by the lead actor.",
        "Complete waste of time.",
        "A masterpiece of modern cinema.",
        "Boring and predictable plot.",
    ] * 30  # Repeat to get more training examples

    labels = [1, 0, 1, 0, 1, 0] * 30

    # Tokenize
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    label_tensor = torch.tensor(labels)

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(input_ids, label_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16)

    return model, train_loader, val_loader


def main():
    print("=" * 55)
    print("HuggingFace Classifier Fine-tuning Demo")
    print("=" * 55)

    # ── Set up experiment tracker ──────────────────────────────────────────────
    tracker = ExperimentTracker(log_dir="./experiments")
    run_name = "finetune_v1"

    config = {
        "model": "distilbert-base-uncased" if _HF_AVAILABLE else "dummy_mlp",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "task": "text-classification",
    }

    tracker.start_run(run_name, config=config)
    print(f"\nStarted experiment run: {run_name}")
    print(f"Config: {config}")

    # ── Build model and dataloaders ────────────────────────────────────────────
    if _HF_AVAILABLE:
        try:
            model, train_loader, val_loader = finetune_with_hf()
            loss_fn = nn.CrossEntropyLoss()

            # For HF models, we need a wrapper since their forward returns a SequenceClassifierOutput
            class HFModelWrapper(nn.Module):
                def __init__(self, hf_model):
                    super().__init__()
                    self.model = hf_model

                def forward(self, input_ids):
                    output = self.model(input_ids=input_ids)
                    return output.logits

            model = HFModelWrapper(model)
        except Exception as e:
            print(f"HuggingFace loading failed ({e}), falling back to dummy model")
            _HF_AVAILABLE_LOCAL = False
    else:
        _HF_AVAILABLE_LOCAL = False

    if not _HF_AVAILABLE or '_HF_AVAILABLE_LOCAL' in dir() and not _HF_AVAILABLE_LOCAL:
        print("\nUsing synthetic DummyClassifier for demonstration...")
        model = DummyClassifier(input_dim=64, num_classes=2)
        train_loader = make_dummy_dataloader(200, 64, 2)
        val_loader = make_dummy_dataloader(50, 64, 2)
        loss_fn = nn.CrossEntropyLoss()

    # ── Set up trainer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    def epoch_callback(state):
        epoch = state["epoch"]
        print(
            f"  Epoch {epoch + 1}: train_loss={state['train_loss']:.4f}  "
            f"train_acc={state['train_accuracy']:.4f}"
            + (f"  val_loss={state['val_loss']:.4f}  val_acc={state['val_accuracy']:.4f}"
               if state.get("val_loss") is not None else "")
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        tracker=tracker,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\nTraining for {config['epochs']} epochs...")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config["epochs"],
        callbacks=[epoch_callback],
    )

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\nFinal evaluation:")
    eval_result = trainer.evaluate(val_loader)
    print(f"  val_loss={eval_result.loss:.4f}  val_accuracy={eval_result.accuracy:.4f}")
    print(f"  latency={eval_result.latency_ms_per_batch:.2f}ms/batch")

    # ── Log artifact and end run ───────────────────────────────────────────────
    tracker.log_artifact("checkpoints/model_final.pt")  # Would save here in real workflow
    tracker.end_run()

    print(f"\nRun '{run_name}' completed. Logs written to ./experiments/runs.jsonl")
    print(f"Best epoch: {history.best_epoch + 1} (val_loss={history.best_val_loss:.4f})")


if __name__ == "__main__":
    main()

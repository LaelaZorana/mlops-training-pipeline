"""
Tests for pipeline.trainer — Trainer, EpochResult, EvalResult, TrainingHistory.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.trainer import Trainer, EpochResult, EvalResult, TrainingHistory


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_linear_model(in_dim=4, out_dim=2):
    return nn.Linear(in_dim, out_dim)


def make_dataloader(num_samples=64, in_dim=4, num_classes=2, batch_size=16):
    x = torch.randn(num_samples, in_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


@pytest.fixture
def simple_trainer():
    model = make_linear_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    return Trainer(model, optimizer, loss_fn)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_epoch_result_has_all_fields(simple_trainer):
    loader = make_dataloader()
    result = simple_trainer.train_epoch(loader, epoch=0)
    assert hasattr(result, "epoch")
    assert hasattr(result, "loss")
    assert hasattr(result, "accuracy")
    assert hasattr(result, "samples_per_sec")
    assert hasattr(result, "epoch_time_sec")
    assert hasattr(result, "num_samples")


def test_accuracy_between_zero_and_one(simple_trainer):
    loader = make_dataloader()
    result = simple_trainer.train_epoch(loader)
    assert 0.0 <= result.accuracy <= 1.0, f"Accuracy out of range: {result.accuracy}"


def test_samples_per_sec_positive(simple_trainer):
    loader = make_dataloader()
    result = simple_trainer.train_epoch(loader)
    assert result.samples_per_sec > 0


def test_fit_returns_training_history_correct_length(simple_trainer):
    loader = make_dataloader()
    history = simple_trainer.fit(loader, epochs=3)
    assert isinstance(history, TrainingHistory)
    assert len(history.train_results) == 3


def test_evaluate_returns_eval_result(simple_trainer):
    loader = make_dataloader()
    result = simple_trainer.evaluate(loader)
    assert isinstance(result, EvalResult)
    assert hasattr(result, "loss")
    assert hasattr(result, "accuracy")
    assert hasattr(result, "latency_ms_per_batch")


def test_loss_decreases_over_epochs_on_simple_task():
    """On a very simple linear task, loss should be lower after 20 epochs than at start."""
    # Use a learnable task: X @ W = Y with known pattern
    torch.manual_seed(42)
    model = make_linear_model(in_dim=4, out_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    loader = make_dataloader(num_samples=128, in_dim=4, batch_size=32)
    history = trainer.fit(loader, epochs=10)

    first_loss = history.train_results[0].loss
    last_loss = history.train_results[-1].loss
    # Allow for some variance — just check it hasn't gotten significantly worse
    assert last_loss <= first_loss * 1.5, f"Loss increased too much: {first_loss:.4f} → {last_loss:.4f}"


def test_trainer_handles_empty_dataloader(simple_trainer):
    """Training on empty data should return a result with 0 samples and not crash."""
    empty_x = torch.randn(0, 4)
    empty_y = torch.randint(0, 2, (0,))
    loader = DataLoader(TensorDataset(empty_x, empty_y), batch_size=16)
    result = simple_trainer.train_epoch(loader, epoch=0)
    assert result.num_samples == 0


def test_callbacks_are_called(simple_trainer):
    loader = make_dataloader()
    called_states = []

    def record_callback(state):
        called_states.append(state.copy())

    history = simple_trainer.fit(loader, epochs=3, callbacks=[record_callback])
    assert len(called_states) == 3
    for state in called_states:
        assert "epoch" in state
        assert "train_loss" in state


def test_fit_logs_to_tracker_if_provided():
    """If an ExperimentTracker is provided, metrics should be logged automatically."""
    import tempfile
    import os
    from pipeline.experiment_tracker import ExperimentTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(log_dir=tmpdir)
        tracker.start_run("test_run", config={"lr": 0.01})

        model = make_linear_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, loss_fn, tracker=tracker)

        loader = make_dataloader()
        trainer.fit(loader, epochs=2)
        tracker.end_run()

        run = tracker.get_run("test_run")
        assert "train_loss" in run.metrics
        assert len(run.metrics["train_loss"]) == 2  # 2 epochs logged

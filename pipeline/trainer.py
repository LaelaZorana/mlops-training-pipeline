"""
Training loop abstraction that works with PyTorch models and is
duck-typing compatible with JAX functional interfaces.

The Trainer class owns:
  - train_epoch: one full pass over training data
  - evaluate: full evaluation pass with latency measurement
  - fit: multi-epoch loop with callbacks and optional tracker integration

Design choices:
  - No coupling to specific model architectures, just needs model(inputs) → outputs
  - Callbacks receive a dict of current state (epoch, loss, accuracy, etc.)
  - Works on CPU by default; put model + data on CUDA before passing in
  - Duck-typing compatible: any object with .parameters() and forward() works
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class EpochResult:
    """
    Summary of one training epoch.

    Attributes:
        epoch: 0-indexed epoch number.
        loss: Mean training loss over the epoch.
        accuracy: Mean training accuracy [0, 1].
        samples_per_sec: Training throughput in samples/second.
        epoch_time_sec: Wall-clock time for the epoch.
        num_samples: Total samples processed.
    """
    epoch: int
    loss: float
    accuracy: float
    samples_per_sec: float
    epoch_time_sec: float
    num_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "samples_per_sec": self.samples_per_sec,
            "epoch_time_sec": self.epoch_time_sec,
            "num_samples": self.num_samples,
        }


@dataclass
class EvalResult:
    """
    Evaluation pass results.

    Attributes:
        loss: Mean evaluation loss.
        accuracy: Mean evaluation accuracy [0, 1].
        latency_ms_per_batch: Average inference latency per batch in ms.
        num_samples: Total evaluation samples.
    """
    loss: float
    accuracy: float
    latency_ms_per_batch: float
    num_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "latency_ms_per_batch": self.latency_ms_per_batch,
            "num_samples": self.num_samples,
        }


@dataclass
class TrainingHistory:
    """
    Full training run history across all epochs.

    Attributes:
        train_results: List of EpochResult, one per epoch.
        val_results: List of EvalResult, one per epoch (if validation was run).
        best_epoch: Index of the epoch with the lowest val loss (or last if no val).
        best_val_loss: Lowest validation loss seen.
    """
    train_results: List[EpochResult] = field(default_factory=list)
    val_results: List[Optional[EvalResult]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")

    def to_dict(self) -> dict:
        return {
            "train_results": [r.to_dict() for r in self.train_results],
            "val_results": [r.to_dict() if r else None for r in self.val_results],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "num_epochs": len(self.train_results),
        }


class Trainer:
    """
    Generic training loop for PyTorch models (or any duck-type equivalent).

    Supports:
      - CPU and CUDA training
      - Epoch-level callbacks
      - Optional ExperimentTracker integration
      - Validation after each epoch

    Usage:
        import torch
        import torch.nn as nn

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        trainer = Trainer(model, optimizer, loss_fn)
        history = trainer.fit(train_loader, val_loader, epochs=5)
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        device: str = "cpu",
        tracker: Optional[Any] = None,
    ):
        """
        Args:
            model: PyTorch model (or duck-type with __call__ + parameters()).
            optimizer: Optimizer (e.g., torch.optim.Adam).
            loss_fn: Loss function (e.g., nn.CrossEntropyLoss()).
            device: Device string, e.g. 'cpu', 'cuda', 'cuda:0'.
            tracker: Optional ExperimentTracker instance for logging metrics.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.tracker = tracker

    def train_epoch(self, dataloader: Any, epoch: int = 0) -> EpochResult:
        """
        Run one full training epoch.

        Args:
            dataloader: Iterable yielding (inputs, targets) batches.
            epoch: Current epoch index (0-based), used for logging.

        Returns:
            EpochResult with loss, accuracy, throughput, and timing.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for Trainer. pip install torch")

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        t_start = time.perf_counter()

        for batch_inputs, batch_targets in dataloader:
            # Move to device
            if hasattr(batch_inputs, "to"):
                batch_inputs = batch_inputs.to(self.device)
            if hasattr(batch_targets, "to"):
                batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_inputs)
            loss = self.loss_fn(outputs, batch_targets)
            loss.backward()
            self.optimizer.step()

            # Accumulate stats
            total_loss += loss.item()
            batch_size = batch_inputs.shape[0] if hasattr(batch_inputs, "shape") else 1
            total_samples += batch_size

            # Compute accuracy (works for classification outputs)
            if outputs.ndim == 2:
                predicted = outputs.argmax(dim=1)
                correct = (predicted == batch_targets).sum().item()
                total_correct += correct

            num_batches += 1

        epoch_time = time.perf_counter() - t_start

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_samples, 1)
        samples_per_sec = total_samples / max(epoch_time, 1e-9)

        return EpochResult(
            epoch=epoch,
            loss=avg_loss,
            accuracy=accuracy,
            samples_per_sec=samples_per_sec,
            epoch_time_sec=epoch_time,
            num_samples=total_samples,
        )

    def evaluate(self, dataloader: Any) -> EvalResult:
        """
        Run evaluation (no gradient computation).

        Args:
            dataloader: Iterable yielding (inputs, targets) batches.

        Returns:
            EvalResult with loss, accuracy, and per-batch latency.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for Trainer. pip install torch")

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        batch_latencies_ms = []

        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                if hasattr(batch_inputs, "to"):
                    batch_inputs = batch_inputs.to(self.device)
                if hasattr(batch_targets, "to"):
                    batch_targets = batch_targets.to(self.device)

                t0 = time.perf_counter()
                outputs = self.model(batch_inputs)
                t1 = time.perf_counter()

                batch_latencies_ms.append((t1 - t0) * 1000.0)
                loss = self.loss_fn(outputs, batch_targets)
                total_loss += loss.item()

                batch_size = batch_inputs.shape[0] if hasattr(batch_inputs, "shape") else 1
                total_samples += batch_size

                if outputs.ndim == 2:
                    predicted = outputs.argmax(dim=1)
                    correct = (predicted == batch_targets).sum().item()
                    total_correct += correct

                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_samples, 1)
        avg_latency = sum(batch_latencies_ms) / max(len(batch_latencies_ms), 1)

        return EvalResult(
            loss=avg_loss,
            accuracy=accuracy,
            latency_ms_per_batch=avg_latency,
            num_samples=total_samples,
        )

    def fit(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 10,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingHistory:
        """
        Run the full training loop for N epochs.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader (optional).
            epochs: Number of epochs to train.
            callbacks: List of callables; each receives a state dict after each epoch.
                       State dict keys: epoch, train_loss, train_accuracy,
                       val_loss (if val_loader provided), val_accuracy.

        Returns:
            TrainingHistory with all epoch results.
        """
        callbacks = callbacks or []
        history = TrainingHistory()

        for epoch in range(epochs):
            epoch_result = self.train_epoch(train_loader, epoch=epoch)
            history.train_results.append(epoch_result)

            val_result = None
            if val_loader is not None:
                val_result = self.evaluate(val_loader)

                if val_result.loss < history.best_val_loss:
                    history.best_val_loss = val_result.loss
                    history.best_epoch = epoch

            history.val_results.append(val_result)

            # Log to tracker if provided
            if self.tracker is not None:
                step = epoch
                self.tracker.log_metrics({
                    "train_loss": epoch_result.loss,
                    "train_accuracy": epoch_result.accuracy,
                    "samples_per_sec": epoch_result.samples_per_sec,
                }, step=step)
                if val_result:
                    self.tracker.log_metrics({
                        "val_loss": val_result.loss,
                        "val_accuracy": val_result.accuracy,
                    }, step=step)

            # Call user-provided callbacks
            state = {
                "epoch": epoch,
                "train_loss": epoch_result.loss,
                "train_accuracy": epoch_result.accuracy,
                "val_loss": val_result.loss if val_result else None,
                "val_accuracy": val_result.accuracy if val_result else None,
                "history": history,
            }
            for cb in callbacks:
                cb(state)

        return history

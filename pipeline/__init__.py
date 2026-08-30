"""
mlops-training-pipeline: Composable ML training infrastructure.

Modules:
  trainer              - Training loop abstraction (PyTorch + JAX-compatible)
  experiment_tracker   - Local JSONL-backed experiment tracking
  model_evaluator      - HuggingFace model evaluation and inference benchmarking
  jax_trainer          - JAX functional training utilities
  distributed_config   - Distributed training configuration and validation
"""

from pipeline.trainer import Trainer, EpochResult, EvalResult, TrainingHistory
from pipeline.experiment_tracker import ExperimentTracker, RunRecord, ComparisonReport
from pipeline.distributed_config import DistributedConfig

__version__ = "0.1.0"
__all__ = [
    "Trainer",
    "EpochResult",
    "EvalResult",
    "TrainingHistory",
    "ExperimentTracker",
    "RunRecord",
    "ComparisonReport",
    "DistributedConfig",
]

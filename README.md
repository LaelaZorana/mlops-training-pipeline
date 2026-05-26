# mlops-training-pipeline

I kept seeing ML teams reinvent the same training infrastructure from scratch on every new project — experiment tracking bolted on as an afterthought, no standard way to configure distributed training, evaluation logic copy-pasted between repos. I wanted a clean, composable toolkit that handles the common patterns (training loops, experiment tracking, HuggingFace evaluation, JAX support, distributed config) in one place, without locking into the heavyweight MLflow/Weights & Biases setup that's often overkill for research iteration.

This is that toolkit. Five focused modules, each doing one thing well, all designed to compose without fighting each other.

---

## What It Does

### `Trainer` — Training Loop Abstraction
Wraps a PyTorch model, optimizer, and loss function. `train_epoch()` handles the forward-backward-step loop and returns an `EpochResult` with loss, accuracy, samples/sec, and timing. `fit()` runs the full multi-epoch loop with optional validation and callbacks. Automatically logs to `ExperimentTracker` if you pass one in.

### `ExperimentTracker` — Local JSONL-backed Tracking
No MLflow server, no W&B account needed. Writes every metric and artifact as a JSON line to a local file. `start_run()` → `log_metrics()` → `end_run()`. `compare_runs()` identifies the best run by any metric. JSONL format means experiments are portable and auditable with a text editor.

### `ModelEvaluator` — HuggingFace Model Evaluation
`evaluate_hf_model()` loads any HuggingFace model, runs it on a dataset, and returns accuracy, F1, per-sample latency, throughput, and model size. `benchmark_inference()` benchmarks any callable for p50/p95/p99 latency. `compare_models()` gives a recommendation based on accuracy/latency tradeoff.

### `JAXTrainer` — JAX Functional Training Utilities
JAX-compatible training primitives that follow the functional paradigm: explicit PRNG keys, immutable `TrainState` updates, `jax.jit`-compiled train steps. Detailed comments explaining the key differences from PyTorch's imperative approach.

### `DistributedConfig` — Distributed Training Configuration
Declarative config for DataParallel, DDP, FSDP, and ModelParallel. `validate()` catches incompatible settings (e.g., DataParallel across nodes). `estimate_memory_gb()` gives per-device memory estimates for sizing your cluster. `to_torch_config()` and `to_jax_config()` produce framework-ready dicts.

---

## Quick Start

```bash
pip install torch pytest numpy
# Optional: pip install transformers datasets jax jaxlib optax flax
git clone https://github.com/LaelaZorana/mlops-training-pipeline
cd mlops-training-pipeline
```

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pipeline.trainer import Trainer
from pipeline.experiment_tracker import ExperimentTracker

# Model + data
model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(200, 10)
y = torch.randint(0, 2, (200,))
loader = DataLoader(TensorDataset(x, y), batch_size=32)

# Track the experiment
tracker = ExperimentTracker(log_dir="./experiments")
tracker.start_run("baseline", config={"lr": 1e-3, "epochs": 5})

trainer = Trainer(model, optimizer, loss_fn, tracker=tracker)
history = trainer.fit(loader, epochs=5)

tracker.end_run()
print(f"Best epoch: {history.best_epoch}")
```

---

## HuggingFace Model Evaluation

```python
from pipeline.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()
report = evaluator.evaluate_hf_model(
    "distilbert-base-uncased-finetuned-sst-2-english",
    dataset="sst2",
    task="text-classification",
)
print(report.summary())
# distilbert-base-uncased-finetuned-sst-2-english [text-classification]
#   Accuracy: 0.9130  F1: 0.9128
#   Latency: 2.41ms/sample  Throughput: 414.9 samples/s
#   Model size: 255.4 MB  Samples: 500
```

---

## JAX Training Example

```python
from pipeline.jax_trainer import init_mlp_params, train_step, train_loop
# See examples/jax_mnist.py for the full working example
```

---

## Distributed Config Example

```python
from pipeline.distributed_config import DistributedConfig, DistributedStrategy, MixedPrecision

# FSDP for a 7B model on 8 GPUs
config = DistributedConfig(
    strategy=DistributedStrategy.FSDP,
    num_gpus=8,
    mixed_precision=MixedPrecision.BF16,
    gradient_checkpointing=True,
)
assert config.is_valid()
print(f"Memory/GPU: {config.estimate_memory_gb(7_000, batch_size=4):.1f} GB")
# Memory/GPU: 18.4 GB
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests require PyTorch. JAX tests require jax + optax. HuggingFace tests require transformers + datasets. All optional deps are skipped gracefully.

---

## Project Layout

```
mlops-training-pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── __main__.py          # CLI
│   ├── trainer.py           # Trainer + EpochResult + TrainingHistory
│   ├── experiment_tracker.py # ExperimentTracker + RunRecord
│   ├── model_evaluator.py   # ModelEvaluator + EvalReport
│   ├── jax_trainer.py       # JAX functional training utilities
│   └── distributed_config.py # DistributedConfig + validation
├── examples/
│   ├── finetune_classifier.py
│   ├── jax_mnist.py
│   └── distributed_setup.py
├── tests/
│   ├── test_trainer.py
│   ├── test_experiment_tracker.py
│   └── test_distributed_config.py
└── requirements.txt
```

---

## Notes

- GPU optional: All modules degrade gracefully on CPU. Full performance requires CUDA.
- JAX optional: `jax_trainer.py` is importable without JAX installed; errors on first use.
- HuggingFace optional: `transformers` and `datasets` only needed for `ModelEvaluator.evaluate_hf_model()`.

---

## License

MIT — Laela Zorana

---

**Links:** [GitHub](https://github.com/LaelaZorana) · [HuggingFace](https://huggingface.co/LaelaZ) · [Kaggle](https://www.kaggle.com/laelazorana)

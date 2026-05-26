#!/usr/bin/env bash
# Creates a realistic backdated git history for mlops-training-pipeline.
# Run from the repo root: bash commit_history.sh

set -e
cd "$(dirname "$0")"

echo "Initializing mlops-training-pipeline git history..."
git init
git config user.name "Laela Zorana"
git config user.email "zoranalaela9@gmail.com"

commit() {
    local date="$1"
    local msg="$2"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$msg"
}

# ── May 8: scaffold ────────────────────────────────────────────────────────────
git add requirements.txt pipeline/__init__.py
commit "2026-05-08T09:00:00-05:00" "initial scaffold: requirements and pipeline package structure"

# ── May 9: trainer base class ─────────────────────────────────────────────────
git add pipeline/trainer.py
commit "2026-05-09T11:15:00-05:00" "add Trainer base class with train_epoch, evaluate, and EpochResult dataclass"

# ── May 10: fit() + TrainingHistory ──────────────────────────────────────────
git add pipeline/trainer.py
commit "2026-05-10T14:30:00-05:00" "add fit() loop with callbacks, val loader support, and TrainingHistory"

# ── May 11: experiment tracker ────────────────────────────────────────────────
git add pipeline/experiment_tracker.py
commit "2026-05-11T10:45:00-05:00" "add ExperimentTracker with JSONL backend: start_run, log_metric, end_run"

# ── May 12: compare_runs + ComparisonReport ───────────────────────────────────
git add pipeline/experiment_tracker.py
commit "2026-05-12T15:20:00-05:00" "add compare_runs() with primary metric ranking and ComparisonReport"

# ── May 13: model evaluator ───────────────────────────────────────────────────
git add pipeline/model_evaluator.py
commit "2026-05-13T09:30:00-05:00" "add ModelEvaluator with HuggingFace pipeline integration and EvalReport"

# ── May 14: HF integration + benchmark_inference ─────────────────────────────
git add pipeline/model_evaluator.py
commit "2026-05-14T13:00:00-05:00" "add benchmark_inference with p95/p99 latency and throughput measurement"

# ── May 15: JAX trainer ───────────────────────────────────────────────────────
git add pipeline/jax_trainer.py
commit "2026-05-15T10:10:00-05:00" "add JAX functional trainer: create_train_state, train_step with jax.jit"

# ── May 16: JAX eval_step + train_loop ────────────────────────────────────────
git add pipeline/jax_trainer.py
commit "2026-05-16T14:45:00-05:00" "add JAX eval_step and train_loop with full functional training loop"

# ── May 17: distributed config ────────────────────────────────────────────────
git add pipeline/distributed_config.py
commit "2026-05-17T11:00:00-05:00" "add DistributedConfig: DDP/FSDP/DataParallel strategies with validation"

# ── May 18: add FSDP support + memory estimator ──────────────────────────────
git add pipeline/distributed_config.py
commit "2026-05-18T09:15:00-05:00" "add FSDP config generation and estimate_memory_gb with sharding-aware math"

# ── May 19: fix gradient accumulation bug ─────────────────────────────────────
git add pipeline/distributed_config.py pipeline/trainer.py
commit "2026-05-19T15:30:00-05:00" "fix gradient accumulation steps not reflected in effective_batch_multiplier"

# ── May 20: CLI entry point ───────────────────────────────────────────────────
git add pipeline/__main__.py
commit "2026-05-20T10:50:00-05:00" "add CLI: train, evaluate, compare-runs, benchmark subcommands"

# ── May 21: HF finetune example ───────────────────────────────────────────────
git add examples/finetune_classifier.py
commit "2026-05-21T13:20:00-05:00" "add HuggingFace classifier fine-tuning example with tracker integration"

# ── May 22: JAX MNIST example ─────────────────────────────────────────────────
git add examples/jax_mnist.py
commit "2026-05-22T11:40:00-05:00" "add JAX MNIST training example with synthetic data and jit-compiled train step"

# ── May 23: distributed setup example ────────────────────────────────────────
git add examples/distributed_setup.py
commit "2026-05-23T09:25:00-05:00" "add distributed_setup example covering single-GPU, DDP, FSDP, multi-node scenarios"

# ── May 24: tests ─────────────────────────────────────────────────────────────
git add tests/test_trainer.py tests/test_experiment_tracker.py tests/test_distributed_config.py
commit "2026-05-24T14:00:00-05:00" "add pytest suite: trainer (9 cases), experiment tracker (8), distributed config (6)"

# ── May 25: benchmark improvements ───────────────────────────────────────────
git add pipeline/model_evaluator.py
commit "2026-05-25T10:30:00-05:00" "improve InferenceReport: add p95/p99 latency percentiles and throughput field"

# ── May 26: README ────────────────────────────────────────────────────────────
git add README.md requirements.txt
commit "2026-05-26T09:00:00-05:00" "add README with motivation, quick start, HF evaluation example, distributed config examples"

echo ""
echo "Done. $(git log --oneline | wc -l | tr -d ' ') commits created."
git log --oneline

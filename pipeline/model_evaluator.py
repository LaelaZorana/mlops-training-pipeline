"""
Model evaluation and inference benchmarking.

Supports HuggingFace transformers models for standard NLP tasks and
provides inference latency/throughput benchmarking for any callable model.

HuggingFace dependencies are imported with graceful fallback — this module
can be imported even if transformers/datasets are not installed.
"""

import time
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvalReport:
    """
    Evaluation results for a HuggingFace model on a dataset.

    Attributes:
        model_name: Model name or path.
        task: Task type (e.g., 'text-classification').
        accuracy: Accuracy on the evaluation set [0, 1].
        f1: F1 score (macro average). -1.0 if not computed.
        latency_ms: Mean inference latency per sample in ms.
        throughput_samples_per_sec: Samples processed per second.
        model_size_mb: Model parameter size in MB (float32 equivalent).
        num_samples: Number of evaluation samples.
    """
    model_name: str
    task: str
    accuracy: float
    f1: float
    latency_ms: float
    throughput_samples_per_sec: float
    model_size_mb: float
    num_samples: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "latency_ms": self.latency_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "model_size_mb": self.model_size_mb,
            "num_samples": self.num_samples,
        }

    def summary(self) -> str:
        return (
            f"{self.model_name} [{self.task}]\n"
            f"  Accuracy: {self.accuracy:.4f}  F1: {self.f1:.4f}\n"
            f"  Latency: {self.latency_ms:.2f}ms/sample  "
            f"Throughput: {self.throughput_samples_per_sec:.1f} samples/s\n"
            f"  Model size: {self.model_size_mb:.1f} MB  Samples: {self.num_samples}"
        )


@dataclass
class InferenceReport:
    """
    Inference benchmarking results for any model callable.

    Attributes:
        mean_latency_ms: Average per-call latency in ms.
        p95_latency_ms: 95th percentile latency in ms.
        p99_latency_ms: 99th percentile latency in ms.
        throughput_calls_per_sec: Calls per second.
        iterations: Number of timed calls.
    """
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_calls_per_sec: float
    iterations: int

    def to_dict(self) -> dict:
        return {
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_calls_per_sec": self.throughput_calls_per_sec,
            "iterations": self.iterations,
        }


@dataclass
class ModelComparisonReport:
    """
    Side-by-side comparison of two EvalReports.

    Attributes:
        model_a: Name of the first model.
        model_b: Name of the second model.
        accuracy_delta: model_b.accuracy - model_a.accuracy.
        latency_speedup_x: model_a.latency_ms / model_b.latency_ms (higher = model_b is faster).
        size_ratio: model_b.model_size_mb / model_a.model_size_mb.
        recommendation: Which model is preferred and why.
    """
    model_a: str
    model_b: str
    accuracy_delta: float
    latency_speedup_x: float
    size_ratio: float
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "accuracy_delta": self.accuracy_delta,
            "latency_speedup_x": self.latency_speedup_x,
            "size_ratio": self.size_ratio,
            "recommendation": self.recommendation,
        }


def _get_model_size_mb(model: Any) -> float:
    """Estimate model parameter size in MB (assumes float32)."""
    try:
        import torch
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    except Exception:
        return -1.0


def _percentile(data: list, pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class ModelEvaluator:
    """
    Evaluates HuggingFace models and benchmarks inference latency/throughput.

    Usage:
        evaluator = ModelEvaluator()

        # Evaluate a HuggingFace model
        report = evaluator.evaluate_hf_model(
            "distilbert-base-uncased-finetuned-sst-2-english",
            dataset="sst2",
            task="text-classification"
        )
        print(report.summary())

        # Benchmark any callable
        inf_report = evaluator.benchmark_inference(my_model, dummy_input)
        print(f"p99 latency: {inf_report.p99_latency_ms:.2f}ms")
    """

    def evaluate_hf_model(
        self,
        model_name_or_path: str,
        dataset: Any,
        task: str = "text-classification",
        max_samples: int = 500,
        device: str = "cpu",
    ) -> EvalReport:
        """
        Evaluate a HuggingFace model on a dataset.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            dataset: Dataset name (str) or HuggingFace Dataset object.
            task: Task type. Supported: 'text-classification'.
            max_samples: Maximum samples to evaluate (for speed).
            device: Device to run inference on ('cpu', 'cuda').

        Returns:
            EvalReport with accuracy, F1, latency, and throughput.

        Raises:
            ImportError: If transformers/datasets are not installed.
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "transformers not installed. pip install transformers\n"
                "Also recommended: pip install datasets"
            )

        # Load dataset
        if isinstance(dataset, str):
            try:
                from datasets import load_dataset
                ds = load_dataset(dataset, split="validation")
            except ImportError:
                raise ImportError("datasets not installed. pip install datasets")
        else:
            ds = dataset

        # Limit samples
        if hasattr(ds, "select"):
            num = min(max_samples, len(ds))
            ds = ds.select(range(num))

        # Build pipeline
        pipe = hf_pipeline(task, model=model_name_or_path, device=0 if device == "cuda" else -1)

        # Get model size
        model_size_mb = _get_model_size_mb(pipe.model)

        # Run evaluation
        texts = [ex.get("sentence", ex.get("text", "")) for ex in ds]
        labels = [ex.get("label", 0) for ex in ds]

        t_start = time.perf_counter()
        predictions = pipe(texts, batch_size=32)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        throughput = len(texts) / elapsed
        latency_ms = elapsed / len(texts) * 1000.0

        # Compute accuracy
        label_map = pipe.model.config.label2id if hasattr(pipe.model.config, "label2id") else {}
        correct = 0
        pred_labels = []
        for pred, true_label in zip(predictions, labels):
            pred_label_str = pred["label"]
            pred_id = label_map.get(pred_label_str, -1) if label_map else -1
            if pred_id == true_label:
                correct += 1
            pred_labels.append(pred_id)

        accuracy = correct / len(texts) if texts else 0.0

        # Compute F1 (macro)
        f1_score = -1.0
        try:
            from sklearn.metrics import f1_score
            f1_score_val = f1_score(labels, pred_labels, average="macro", zero_division=0)
            f1_score = float(f1_score_val)
        except ImportError:
            pass  # sklearn not available

        return EvalReport(
            model_name=model_name_or_path,
            task=task,
            accuracy=accuracy,
            f1=f1_score,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput,
            model_size_mb=model_size_mb,
            num_samples=len(texts),
        )

    def benchmark_inference(
        self,
        model: Any,
        inputs: Any,
        iterations: int = 100,
        warmup: int = 10,
    ) -> InferenceReport:
        """
        Benchmark inference latency for any callable model.

        Args:
            model: Any callable that takes inputs.
            inputs: Input(s) to pass to the model on each call.
            iterations: Number of timed iterations.
            warmup: Warm-up iterations before timing.

        Returns:
            InferenceReport with latency distribution and throughput.
        """
        # Warm up
        for _ in range(warmup):
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
            else:
                model(inputs)

        latencies_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
            else:
                model(inputs)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        mean_lat = sum(latencies_ms) / len(latencies_ms)
        p95 = _percentile(latencies_ms, 95)
        p99 = _percentile(latencies_ms, 99)
        total_time = sum(latencies_ms) / 1000.0
        throughput = iterations / total_time if total_time > 0 else 0.0

        return InferenceReport(
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            throughput_calls_per_sec=throughput,
            iterations=iterations,
        )

    def compare_models(
        self,
        model_a_report: EvalReport,
        model_b_report: EvalReport,
    ) -> ModelComparisonReport:
        """
        Compare two EvalReports and produce a recommendation.

        Args:
            model_a_report: Baseline model's EvalReport.
            model_b_report: Candidate model's EvalReport.

        Returns:
            ModelComparisonReport with delta, speedup, size ratio, and recommendation.
        """
        acc_delta = model_b_report.accuracy - model_a_report.accuracy
        lat_speedup = (
            model_a_report.latency_ms / model_b_report.latency_ms
            if model_b_report.latency_ms > 0
            else 1.0
        )
        size_ratio = (
            model_b_report.model_size_mb / model_a_report.model_size_mb
            if model_a_report.model_size_mb > 0
            else 1.0
        )

        # Simple recommendation heuristic
        if acc_delta > 0.02 and lat_speedup >= 1.0:
            rec = f"Prefer {model_b_report.model_name}: higher accuracy (+{acc_delta:.3f}) with equal/better latency."
        elif acc_delta < -0.02:
            rec = f"Prefer {model_a_report.model_name}: {model_b_report.model_name} loses accuracy ({acc_delta:.3f})."
        elif lat_speedup > 1.5:
            rec = f"Prefer {model_b_report.model_name}: {lat_speedup:.1f}x faster with similar accuracy."
        else:
            rec = f"Similar performance. {model_b_report.model_name} is {lat_speedup:.1f}x faster, accuracy delta={acc_delta:.3f}."

        return ModelComparisonReport(
            model_a=model_a_report.model_name,
            model_b=model_b_report.model_name,
            accuracy_delta=acc_delta,
            latency_speedup_x=lat_speedup,
            size_ratio=size_ratio,
            recommendation=rec,
        )

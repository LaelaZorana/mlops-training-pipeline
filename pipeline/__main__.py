"""
CLI entry point for mlops-training-pipeline.

Usage:
    python -m pipeline train --config config.yaml
    python -m pipeline evaluate --model <path_or_hf_name> --dataset <dataset_name>
    python -m pipeline compare-runs run1 run2 [run3 ...]
    python -m pipeline benchmark --model <hf_model_name>
"""

import sys
import argparse


def cmd_train(args):
    """Train a model from a YAML config file."""
    import yaml

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    except ImportError:
        print("PyYAML not installed. pip install pyyaml")
        sys.exit(1)

    print(f"Training config: {args.config}")
    print(f"Config contents: {config}")

    # In a real workflow: build model, dataloaders, trainer from config
    # For now, print the config and indicate what would happen
    print("\nTo use this CLI:")
    print("  1. Define your model/dataset/training config in a YAML file")
    print("  2. Import and instantiate Trainer from pipeline.trainer")
    print("  3. Call trainer.fit(train_loader, val_loader, epochs=config['epochs'])")


def cmd_evaluate(args):
    """Evaluate a HuggingFace model on a dataset."""
    from pipeline.model_evaluator import ModelEvaluator

    print(f"Evaluating model: {args.model}")
    print(f"Dataset: {args.dataset}")

    evaluator = ModelEvaluator()
    try:
        report = evaluator.evaluate_hf_model(
            args.model,
            dataset=args.dataset,
            task=getattr(args, "task", "text-classification"),
        )
        print(report.summary())
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)


def cmd_compare_runs(args):
    """Compare two or more experiment runs."""
    from pipeline.experiment_tracker import ExperimentTracker

    log_dir = getattr(args, "log_dir", "./experiments")
    tracker = ExperimentTracker(log_dir=log_dir)

    try:
        report = tracker.compare_runs(
            args.run_names,
            primary_metric=getattr(args, "metric", "val_loss"),
        )
        print(report.summary)
    except KeyError as e:
        print(f"Run not found: {e}")
        sys.exit(1)


def cmd_benchmark(args):
    """Benchmark inference latency for a HuggingFace model."""
    print(f"Benchmarking: {args.model}")

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        model.eval()

        # Create dummy input
        dummy_text = "This is a benchmark test sentence for measuring inference latency."
        inputs = tokenizer(dummy_text, return_tensors="pt")

        from pipeline.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator()

        def model_call(input_ids, attention_mask):
            with torch.no_grad():
                return model(input_ids=input_ids, attention_mask=attention_mask)

        report = evaluator.benchmark_inference(
            model_call,
            inputs=(inputs["input_ids"], inputs["attention_mask"]),
            iterations=50,
        )
        print(f"Mean latency:   {report.mean_latency_ms:.2f} ms")
        print(f"p95 latency:    {report.p95_latency_ms:.2f} ms")
        print(f"p99 latency:    {report.p99_latency_ms:.2f} ms")
        print(f"Throughput:     {report.throughput_calls_per_sec:.1f} calls/sec")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="MLOps training pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    p_train = subparsers.add_parser("train", help="Train from a YAML config")
    p_train.add_argument("--config", required=True, help="Path to YAML config file")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a HuggingFace model")
    p_eval.add_argument("--model", required=True, help="Model name or path")
    p_eval.add_argument("--dataset", required=True, help="Dataset name")
    p_eval.add_argument("--task", default="text-classification", help="Task type")

    # compare-runs
    p_compare = subparsers.add_parser("compare-runs", help="Compare experiment runs")
    p_compare.add_argument("run_names", nargs="+", help="Run names to compare")
    p_compare.add_argument("--log-dir", default="./experiments")
    p_compare.add_argument("--metric", default="val_loss")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark HuggingFace model inference")
    p_bench.add_argument("--model", required=True, help="HuggingFace model name")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare-runs":
        cmd_compare_runs(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

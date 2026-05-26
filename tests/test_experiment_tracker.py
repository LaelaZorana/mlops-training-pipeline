"""
Tests for pipeline.experiment_tracker — ExperimentTracker, RunRecord, ComparisonReport.
"""

import json
import os
import tempfile
import pytest

from pipeline.experiment_tracker import ExperimentTracker, RunRecord, ComparisonReport


@pytest.fixture
def tracker(tmp_path):
    """Tracker backed by a temp directory."""
    return ExperimentTracker(log_dir=str(tmp_path))


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_start_run_creates_run(tracker):
    tracker.start_run("run_001", config={"lr": 1e-3})
    record = tracker.get_run("run_001")
    assert isinstance(record, RunRecord)
    assert record.name == "run_001"
    assert record.config["lr"] == 1e-3
    assert record.status == "running"


def test_log_metric_stores_value(tracker):
    tracker.start_run("run_002")
    tracker.log_metric("train_loss", 0.85, step=0)
    record = tracker.get_run("run_002")
    assert "train_loss" in record.metrics
    assert len(record.metrics["train_loss"]) == 1
    assert record.metrics["train_loss"][0] == (0, 0.85)


def test_log_metrics_stores_multiple(tracker):
    tracker.start_run("run_003")
    tracker.log_metrics({"val_loss": 0.72, "val_accuracy": 0.81}, step=1)
    record = tracker.get_run("run_003")
    assert "val_loss" in record.metrics
    assert "val_accuracy" in record.metrics


def test_end_run_closes_run(tracker):
    tracker.start_run("run_004")
    tracker.end_run()
    record = tracker.get_run("run_004")
    assert record.status == "completed"
    assert record.end_time is not None


def test_get_run_returns_correct_record(tracker):
    tracker.start_run("run_005", config={"batch_size": 32})
    tracker.log_metric("loss", 1.5, step=0)
    tracker.log_metric("loss", 1.2, step=1)
    tracker.end_run()

    record = tracker.get_run("run_005")
    assert record.config["batch_size"] == 32
    assert len(record.metrics["loss"]) == 2
    assert record.final_metric("loss") == 1.2


def test_compare_runs_identifies_best(tracker):
    # Run A: higher val_loss (worse)
    tracker.start_run("run_a")
    tracker.log_metric("val_loss", 0.9, step=0)
    tracker.log_metric("val_loss", 0.75, step=1)
    tracker.end_run()

    # Run B: lower val_loss (better)
    tracker.start_run("run_b")
    tracker.log_metric("val_loss", 0.8, step=0)
    tracker.log_metric("val_loss", 0.50, step=1)
    tracker.end_run()

    report = tracker.compare_runs(["run_a", "run_b"], primary_metric="val_loss")
    assert isinstance(report, ComparisonReport)
    assert report.best_run == "run_b"


def test_jsonl_file_written_and_readable(tmp_path):
    """The JSONL file should be written and each line should be valid JSON."""
    tracker = ExperimentTracker(log_dir=str(tmp_path))
    tracker.start_run("run_json_test", config={"x": 1})
    tracker.log_metric("loss", 0.5, step=0)
    tracker.end_run()

    log_file = tmp_path / "runs.jsonl"
    assert log_file.exists()

    with open(log_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 3  # start_run + metric + end_run
    for line in lines:
        obj = json.loads(line)  # Should not raise
        assert "event" in obj


def test_nested_metrics_work(tracker):
    """Log multiple metrics across multiple steps."""
    tracker.start_run("run_nested")
    for step in range(5):
        tracker.log_metrics({
            "train_loss": 1.0 - step * 0.1,
            "train_accuracy": 0.5 + step * 0.05,
        }, step=step)
    tracker.end_run()

    record = tracker.get_run("run_nested")
    assert len(record.metrics["train_loss"]) == 5
    assert len(record.metrics["train_accuracy"]) == 5
    assert record.final_metric("train_loss") == pytest.approx(0.6, abs=1e-6)

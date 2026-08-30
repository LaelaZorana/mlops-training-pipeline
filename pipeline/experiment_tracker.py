"""
Local experiment tracker backed by JSONL files.

No external dependencies, no MLflow, no Weights & Biases. Everything lives
in a single JSONL file: one JSON object per log event. This makes runs
portable, reproducible, and auditable without any server infrastructure.

Use this when:
  - You're iterating quickly and don't want to stand up a tracking server
  - You need offline tracking in a CI/CD pipeline
  - You want a simple baseline before deciding whether W&B/MLflow is worth it

Format of each line in the JSONL file:
  {"event": "start_run", "run_name": "...", "config": {...}, "timestamp": "..."}
  {"event": "metric", "run_name": "...", "key": "...", "value": ..., "step": ...}
  {"event": "artifact", "run_name": "...", "path": "...", "timestamp": "..."}
  {"event": "end_run", "run_name": "...", "timestamp": "..."}
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RunRecord:
    """
    Complete record of a single experiment run.

    Attributes:
        name: Unique run identifier.
        config: Hyperparameter/configuration dict provided at run start.
        metrics: Dict of metric_name to list of (step, value) tuples.
        artifacts: List of artifact file paths logged during the run.
        start_time: ISO timestamp when the run started.
        end_time: ISO timestamp when the run ended (None if still active).
        status: 'running', 'completed', or 'failed'.
    """
    name: str
    config: Dict[str, Any]
    metrics: Dict[str, List] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "running"

    def final_metric(self, key: str) -> Optional[float]:
        """Return the last value logged for a metric key."""
        if key not in self.metrics or not self.metrics[key]:
            return None
        return self.metrics[key][-1][1]  # (step, value)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config": self.config,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
        }


@dataclass
class ComparisonReport:
    """
    Side-by-side comparison of multiple experiment runs.

    Attributes:
        run_names: Names of the compared runs.
        best_run: Name of the run with the best primary metric.
        metric_improvements: Dict mapping run_name to improvement vs baseline (first run).
        summary: Human-readable summary string.
    """
    run_names: List[str]
    best_run: str
    metric_improvements: Dict[str, float]
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "run_names": self.run_names,
            "best_run": self.best_run,
            "metric_improvements": self.metric_improvements,
            "summary": self.summary,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentTracker:
    """
    Lightweight experiment tracker that writes to a local JSONL file.

    Usage:
        tracker = ExperimentTracker(log_dir="./experiments")
        tracker.start_run("run_001", config={"lr": 1e-3, "batch_size": 32})
        tracker.log_metric("train_loss", 0.85, step=0)
        tracker.log_metrics({"val_loss": 0.72, "val_acc": 0.81}, step=1)
        tracker.log_artifact("checkpoints/model_epoch1.pt")
        tracker.end_run()

        record = tracker.get_run("run_001")
        print(record.final_metric("val_loss"))
    """

    def __init__(self, log_dir: str = "./experiments"):
        """
        Args:
            log_dir: Directory where the JSONL log file is written.
                     Created if it doesn't exist.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "runs.jsonl")
        self._active_run: Optional[str] = None
        self._runs: Dict[str, RunRecord] = {}
        self._load_existing()

    def _load_existing(self):
        """Load any existing run data from the JSONL file on init."""
        if not os.path.exists(self.log_file):
            return
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    self._apply_event(event)
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupt file: start fresh

    def _apply_event(self, event: dict):
        """Replay a single logged event into the in-memory run registry."""
        name = event.get("run_name", "")
        etype = event.get("event", "")

        if etype == "start_run":
            self._runs[name] = RunRecord(
                name=name,
                config=event.get("config", {}),
                start_time=event.get("timestamp"),
                status="running",
            )
        elif etype == "metric" and name in self._runs:
            key = event["key"]
            value = event["value"]
            step = event.get("step", 0)
            run = self._runs[name]
            if key not in run.metrics:
                run.metrics[key] = []
            run.metrics[key].append((step, value))
        elif etype == "artifact" and name in self._runs:
            self._runs[name].artifacts.append(event.get("path", ""))
        elif etype == "end_run" and name in self._runs:
            self._runs[name].end_time = event.get("timestamp")
            self._runs[name].status = "completed"

    def _write_event(self, event: dict):
        """Append a single event as a JSON line to the log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def start_run(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Start a new experiment run.

        Args:
            name: Unique run identifier. Raises if name already active.
            config: Hyperparameter configuration dict to associate with this run.
        """
        if name in self._runs and self._runs[name].status == "running":
            raise ValueError(f"Run '{name}' is already active. Call end_run() first.")

        config = config or {}
        record = RunRecord(name=name, config=config, start_time=_now_iso())
        self._runs[name] = record
        self._active_run = name

        event = {
            "event": "start_run",
            "run_name": name,
            "config": config,
            "timestamp": record.start_time,
        }
        self._apply_event(event)
        self._write_event(event)

    def log_metric(self, key: str, value: float, step: int = 0):
        """
        Log a single metric value.

        Args:
            key: Metric name (e.g., 'train_loss', 'val_accuracy').
            value: Numeric value.
            step: Training step or epoch index.
        """
        run_name = self._active_run
        if run_name is None:
            raise RuntimeError("No active run. Call start_run() first.")

        event = {
            "event": "metric",
            "run_name": run_name,
            "key": key,
            "value": value,
            "step": step,
            "timestamp": _now_iso(),
        }
        self._apply_event(event)
        self._write_event(event)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dict of metric_name to value.
            step: Step index shared across all metrics in this call.
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, path: str):
        """
        Log an artifact file path (checkpoint, report, etc.).

        Args:
            path: File path to the artifact.
        """
        run_name = self._active_run
        if run_name is None:
            raise RuntimeError("No active run. Call start_run() first.")

        event = {
            "event": "artifact",
            "run_name": run_name,
            "path": path,
            "timestamp": _now_iso(),
        }
        self._apply_event(event)
        self._write_event(event)

    def end_run(self):
        """Mark the active run as completed."""
        run_name = self._active_run
        if run_name is None:
            return  # Nothing to close

        event = {
            "event": "end_run",
            "run_name": run_name,
            "timestamp": _now_iso(),
        }
        self._apply_event(event)
        self._write_event(event)
        self._active_run = None

    def get_run(self, name: str) -> RunRecord:
        """
        Retrieve a run record by name.

        Args:
            name: Run identifier.

        Returns:
            RunRecord for that run.

        Raises:
            KeyError: If no run with that name exists.
        """
        if name not in self._runs:
            raise KeyError(f"No run named '{name}' found.")
        return self._runs[name]

    def compare_runs(
        self,
        run_names: List[str],
        primary_metric: str = "val_loss",
        lower_is_better: bool = True,
    ) -> ComparisonReport:
        """
        Compare a list of runs by a primary metric.

        Args:
            run_names: List of run names to compare.
            primary_metric: Metric key to use for ranking.
            lower_is_better: True for loss metrics, False for accuracy.

        Returns:
            ComparisonReport identifying the best run and relative improvements.
        """
        records = [self.get_run(n) for n in run_names]
        scores = []
        for r in records:
            val = r.final_metric(primary_metric)
            scores.append((r.name, val))

        # Filter out runs that don't have the metric
        valid = [(name, val) for name, val in scores if val is not None]
        if not valid:
            return ComparisonReport(
                run_names=run_names,
                best_run=run_names[0] if run_names else "",
                metric_improvements={},
                summary=f"No runs have metric '{primary_metric}'",
            )

        if lower_is_better:
            best_name, best_val = min(valid, key=lambda x: x[1])
        else:
            best_name, best_val = max(valid, key=lambda x: x[1])

        # Compute improvement vs baseline (first valid run)
        baseline_val = valid[0][1]
        improvements = {}
        for name, val in valid:
            if baseline_val != 0:
                improvements[name] = (baseline_val - val) / abs(baseline_val) * 100.0
            else:
                improvements[name] = 0.0

        lines = [f"Best run: {best_name} ({primary_metric}={best_val:.4f})"]
        for name, val in valid:
            imp = improvements[name]
            lines.append(f"  {name}: {primary_metric}={val:.4f}  ({imp:+.1f}% vs baseline)")

        return ComparisonReport(
            run_names=run_names,
            best_run=best_name,
            metric_improvements=improvements,
            summary="\n".join(lines),
        )

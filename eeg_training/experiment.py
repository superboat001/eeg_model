"""Experiment directory creation, snapshots, logs, and atomic artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "experiment"


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    checkpoints: Path
    metrics: Path
    predictions: Path
    logs: Path
    artifacts: Path
    model_snapshot: Path

    @property
    def status_file(self) -> Path:
        return self.root / "status.json"


def _unique_run_directory(exp_root: Path, experiment_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{_safe_name(experiment_name)}"
    candidate = exp_root / stem
    suffix = 1
    while candidate.exists():
        candidate = exp_root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def create_experiment(
    project_root: Path,
    experiment_name: str,
    original_config: Path,
    resolved_config: Mapping[str, Any],
    data_directory: Path,
    model_source: Path,
    training_sources: Sequence[Path],
) -> ExperimentPaths:
    """Create one run directory and copy all reproducibility inputs into it."""

    exp_root = project_root / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)
    root = _unique_run_directory(exp_root, experiment_name)
    paths = ExperimentPaths(
        root=root,
        checkpoints=root / "checkpoints",
        metrics=root / "metrics",
        predictions=root / "predictions",
        logs=root / "logs",
        artifacts=root / "artifacts",
        model_snapshot=root / "snapshots" / "model_source" / model_source.name,
    )
    for directory in (
        paths.checkpoints,
        paths.metrics,
        paths.predictions,
        paths.logs,
        paths.artifacts,
        paths.model_snapshot.parent,
        root / "config",
        root / "snapshots" / "data_json",
        root / "snapshots" / "training_code",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    def snapshot(source: Path, destination: Path, category: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append(
            {
                "category": category,
                "source": str(source.resolve()),
                "snapshot": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    snapshot(original_config, root / "config" / "input_config.json", "configuration")
    atomic_write_json(root / "config" / "resolved_config.json", resolved_config)
    snapshot(model_source, paths.model_snapshot, "model_source")

    json_files = sorted(path for path in data_directory.rglob("*.json") if path.is_file())
    if not json_files:
        raise FileNotFoundError(
            f"no JSON metadata files found under preprocessing directory: {data_directory}"
        )
    for source in json_files:
        relative = source.relative_to(data_directory)
        snapshot(
            source,
            root / "snapshots" / "data_json" / relative,
            "preprocessing_json",
        )

    for source in training_sources:
        source = source.resolve()
        try:
            relative = source.relative_to(project_root.resolve())
        except ValueError:
            relative = Path(source.name)
        snapshot(
            source,
            root / "snapshots" / "training_code" / relative,
            "training_source",
        )

    atomic_write_json(root / "snapshots" / "manifest.json", manifest)
    atomic_write_json(
        root / "environment.json",
        {
            "captured_at": utc_now(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    )
    atomic_write_json(
        paths.status_file,
        {
            "status": "initializing",
            "created_at": utc_now(),
            "run_directory": str(root),
        },
    )
    return paths


def configure_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"eeg_training.{log_file.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty prediction table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class HistoryWriter:
    """Append each completed epoch to CSV and JSONL immediately."""

    def __init__(self, directory: Path) -> None:
        self.csv_path = directory / "history.csv"
        self.jsonl_path = directory / "history.jsonl"
        self._fieldnames: list[str] | None = None

    def append(self, row: Mapping[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        if list(row.keys()) != self._fieldnames:
            raise ValueError("history row fields changed between epochs")
        csv_exists = self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if not csv_exists:
                writer.writeheader()
            writer.writerow(row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

"""Configuration validation and independent-segment experiment orchestration."""

from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .data import (
    DatasetInfo,
    IndependentSegmentDataset,
    SubjectRecord,
    load_dataset_info,
    segment_class_weights,
    split_manifest,
    stratified_subject_split,
)
from .engine import (
    EpochResult,
    apply_decision_threshold,
    make_grad_scaler,
    make_segment_loader,
    metrics_for_predictions,
    resolve_device,
    run_epoch,
    seed_everything,
)
from .experiment import (
    ExperimentPaths,
    HistoryWriter,
    atomic_torch_save,
    atomic_write_json,
    configure_logger,
    create_experiment,
    utc_now,
    write_predictions,
)
from .metrics import select_balanced_accuracy_threshold
from .modeling import (
    build_graph,
    build_model_components,
    load_model_module,
    model_parameter_summary,
)


DEFAULTS: dict[str, Any] = {
    "schema_version": "3.0",
    "experiment": {
        "name": "eeg_hc_ad_independent_segments",
        "output_root": "exp",
        "seed": 2026,
        "device": "auto",
        "deterministic": True,
        "amp": True,
        "amp_dtype": "float16",
    },
    "data": {
        "directory": "data/eeg/brainlat",
        "metadata_file": "dataset_description.json",
        # Subjects are still the split unit; segments are the model/training unit.
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "split_seed_start": 2026,
        "split_seed_count": 10,
        "batch_size": 64,
        "eval_batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
    },
    "model": {
        "source_file": "model_design/eeg_hc_ad_model.py",
        "config_class": "EEGModelConfig",
        "class_name": "EEGSegmentClassifier",
        "loss_class": "EEGMultiTaskLoss",
        "parameters": {},
        "graph": {
            "type": "standard_montage_knn",
            "montage": "biosemi128",
            "neighbors": 4,
        },
    },
    "loss": {
        "segment_weight": 1.0,
        "physiology_weight": 0.1,
        # No quality labels exist in the current preprocessing output.
        "quality_weight": 0.0,
    },
    "optimizer": {
        "name": "adamw",
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
    },
    "scheduler": {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 2,
        "min_learning_rate": 0.000001,
    },
    "training": {
        "epochs": 30,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "loss_schedule": {
            "classification_warmup_epochs": 5,
            "physiology_ramp_epochs": 5,
            "target_physiology_weight": 0.1,
        },
        "early_stopping": {
            "monitor": "subject_logit_mean_balanced_accuracy",
            "mode": "max",
            "tiebreaker": "segment_loss",
            "start_epoch": 10,
            "patience": 8,
            "min_delta": 0.0,
        },
    },
}


@dataclass(frozen=True)
class PreparedConfiguration:
    project_root: Path
    config_file: Path
    config: dict[str, Any]
    data_info: DatasetInfo
    model_source: Path
    splits: dict[str, list[SubjectRecord]]
    split_description: dict[str, Any]
    split_seed: int


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite_number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _validate_config_values(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    data = config["data"]
    loss = config["loss"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    training = config["training"]
    schedule = training["loss_schedule"]
    early = training["early_stopping"]

    if experiment.get("output_root") != "exp":
        raise ValueError("experiment.output_root is fixed to 'exp' for this project")
    if not str(experiment.get("name", "")).strip():
        raise ValueError("experiment.name cannot be empty")
    _positive_int(experiment["seed"], "experiment.seed", allow_zero=True)
    if experiment["amp_dtype"] not in {"float16", "bfloat16"}:
        raise ValueError("experiment.amp_dtype must be float16 or bfloat16")
    if not isinstance(experiment["deterministic"], bool) or not isinstance(
        experiment["amp"], bool
    ):
        raise ValueError("experiment.deterministic and experiment.amp must be booleans")

    legacy_data_keys = {
        "train_sampling_mode",
        "evaluation_sampling_mode",
        "train_bag_size",
        "eval_bag_size",
        "segments_per_subject_train",
        "segments_per_subject_eval",
    }
    present_legacy_data = sorted(legacy_data_keys.intersection(data))
    if present_legacy_data:
        raise ValueError(
            "bag/sampled data options are incompatible with independent-segment "
            f"training; remove: {', '.join(present_legacy_data)}"
        )
    for name in ("batch_size", "eval_batch_size"):
        _positive_int(data[name], f"data.{name}")
    _positive_int(data["num_workers"], "data.num_workers", allow_zero=True)
    if not isinstance(data["pin_memory"], bool):
        raise ValueError("data.pin_memory must be boolean")
    fractions = [
        _finite_number(data["train_fraction"], "data.train_fraction", 0.0),
        _finite_number(data["validation_fraction"], "data.validation_fraction", 0.0),
        _finite_number(data["test_fraction"], "data.test_fraction", 0.0),
    ]
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("data split fractions must sum to 1")
    _positive_int(data["split_seed_start"], "data.split_seed_start", allow_zero=True)
    _positive_int(data["split_seed_count"], "data.split_seed_count")

    allowed_loss_keys = {"segment_weight", "physiology_weight", "quality_weight"}
    unknown_loss_keys = sorted(set(loss).difference(allowed_loss_keys))
    if unknown_loss_keys:
        raise ValueError(
            "loss contains obsolete/unknown keys for the segment model: "
            + ", ".join(unknown_loss_keys)
        )
    for name in allowed_loss_keys:
        _finite_number(loss[name], f"loss.{name}", 0.0)
    if float(loss["segment_weight"]) == 0.0:
        raise ValueError("loss.segment_weight must be positive")
    if float(loss["quality_weight"]) != 0.0:
        raise ValueError("loss.quality_weight must remain 0 without quality labels")

    if optimizer["name"].lower() != "adamw":
        raise ValueError("optimizer.name currently supports only 'adamw'")
    if _finite_number(optimizer["learning_rate"], "optimizer.learning_rate", 0.0) == 0:
        raise ValueError("optimizer.learning_rate must be positive")
    _finite_number(optimizer["weight_decay"], "optimizer.weight_decay", 0.0)
    betas = optimizer["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(_finite_number(value, "optimizer.betas") >= 1.0 for value in betas)
        or any(value < 0 for value in betas)
    ):
        raise ValueError("optimizer.betas must contain two values in [0, 1)")

    if scheduler["name"].lower() != "reduce_on_plateau":
        raise ValueError("scheduler.name currently supports only 'reduce_on_plateau'")
    factor = _finite_number(scheduler["factor"], "scheduler.factor", 0.0)
    if not 0.0 < factor < 1.0:
        raise ValueError("scheduler.factor must lie in (0, 1)")
    _positive_int(scheduler["patience"], "scheduler.patience", allow_zero=True)
    _finite_number(
        scheduler["min_learning_rate"], "scheduler.min_learning_rate", 0.0
    )

    epochs = _positive_int(training["epochs"], "training.epochs")
    _positive_int(
        training["gradient_accumulation_steps"],
        "training.gradient_accumulation_steps",
    )
    max_grad_norm = training.get("max_grad_norm")
    if max_grad_norm is not None:
        if _finite_number(max_grad_norm, "training.max_grad_norm", 0.0) == 0:
            raise ValueError("training.max_grad_norm must be positive or null")
    warmup = _positive_int(
        schedule["classification_warmup_epochs"],
        "training.loss_schedule.classification_warmup_epochs",
        allow_zero=True,
    )
    ramp = _positive_int(
        schedule["physiology_ramp_epochs"],
        "training.loss_schedule.physiology_ramp_epochs",
    )
    target = _finite_number(
        schedule["target_physiology_weight"],
        "training.loss_schedule.target_physiology_weight",
        0.0,
    )
    if warmup + ramp > epochs:
        raise ValueError("loss warmup plus ramp cannot exceed training.epochs")
    if not math.isclose(target, float(loss["physiology_weight"])):
        raise ValueError(
            "loss.physiology_weight must equal loss_schedule.target_physiology_weight"
        )

    allowed_monitors = {
        "segment_balanced_accuracy",
        "segment_roc_auc_ad",
        "subject_majority_balanced_accuracy",
        "subject_majority_roc_auc_ad",
        "subject_logit_mean_balanced_accuracy",
        "subject_logit_mean_roc_auc_ad",
        "segment_loss",
        "subject_logit_mean_loss",
    }
    monitor = str(early["monitor"])
    if monitor not in allowed_monitors:
        raise ValueError(
            "early_stopping.monitor must be one of: "
            + ", ".join(sorted(allowed_monitors))
        )
    if early["mode"] not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be min or max")
    expected_mode = "min" if monitor.endswith("loss") else "max"
    if early["mode"] != expected_mode:
        raise ValueError(f"{monitor} monitoring must use mode={expected_mode!r}")
    if early.get("tiebreaker") not in {
        None,
        "segment_loss",
        "subject_logit_mean_loss",
    }:
        raise ValueError(
            "early_stopping.tiebreaker must be segment_loss, "
            "subject_logit_mean_loss, or null"
        )
    start_epoch = _positive_int(early["start_epoch"], "early_stopping.start_epoch")
    if start_epoch > epochs:
        raise ValueError("early_stopping.start_epoch cannot exceed training.epochs")
    _positive_int(early["patience"], "early_stopping.patience")
    _finite_number(early["min_delta"], "early_stopping.min_delta", 0.0)


def _prepare_configuration(
    project_root: Path,
    config_file: Path,
    run_name: str | None,
    device_override: str | None,
    split_seed_start_override: int | None = None,
    split_seed_count_override: int | None = None,
    normalize_per_channel_override: bool | None = None,
) -> PreparedConfiguration:
    project_root = project_root.resolve()
    config_file = config_file.resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"training configuration not found: {config_file}")
    try:
        user_config = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid training configuration JSON: {config_file}") from error
    if not isinstance(user_config, dict):
        raise ValueError("training configuration must be a JSON object")
    config = _deep_merge(DEFAULTS, user_config)
    if run_name is not None:
        config["experiment"]["name"] = run_name
    if device_override is not None:
        config["experiment"]["device"] = device_override
    if split_seed_start_override is not None:
        config["data"]["split_seed_start"] = split_seed_start_override
    if split_seed_count_override is not None:
        config["data"]["split_seed_count"] = split_seed_count_override
    if normalize_per_channel_override is not None:
        if not isinstance(normalize_per_channel_override, bool):
            raise ValueError("normalize_per_channel override must be boolean")
        config["model"]["parameters"][
            "normalize_per_channel"
        ] = normalize_per_channel_override
    _validate_config_values(config)
    split_seed = int(config["data"]["split_seed_start"])

    data_directory = _project_path(project_root, str(config["data"]["directory"]))
    metadata_value = Path(str(config["data"]["metadata_file"]))
    metadata_file = (
        metadata_value.resolve()
        if metadata_value.is_absolute()
        else (data_directory / metadata_value).resolve()
    )
    model_source = _project_path(project_root, str(config["model"]["source_file"]))
    if not model_source.is_file():
        raise FileNotFoundError(f"model source file not found: {model_source}")
    data_info = load_dataset_info(data_directory, metadata_file)
    fractions = (
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
        float(config["data"]["test_fraction"]),
    )
    splits = stratified_subject_split(data_info.records, fractions, split_seed)
    description = split_manifest(splits, fractions, split_seed)
    config["resolved"] = {
        "project_root": str(project_root),
        "config_file": str(config_file),
        "data_directory": str(data_directory),
        "metadata_file": str(metadata_file),
        "model_source_file": str(model_source),
        "dataset_name": data_info.dataset_name,
        "n_subjects": len(data_info.records),
        "n_segments": sum(record.n_segments for record in data_info.records),
        "n_channels": data_info.n_channels,
        "n_samples_per_segment": data_info.n_samples,
        "sampling_rate_hz": data_info.sampling_rate,
        "class_mapping": {"HC": 0, "AD": 1},
        "split_unit": "subject",
        "split_seed": split_seed,
        "split_seed_start": int(config["data"]["split_seed_start"]),
        "split_seed_count": int(config["data"]["split_seed_count"]),
        "training_unit": "segment",
        "model_input": "flat independent segments [N, C, T]",
        "subject_aggregations": ["majority_vote", "logit_mean"],
    }
    return PreparedConfiguration(
        project_root=project_root,
        config_file=config_file,
        config=config,
        data_info=data_info,
        model_source=model_source,
        splits=splits,
        split_description=description,
        split_seed=split_seed,
    )


def _segment_sampling_summary(
    records: Sequence[SubjectRecord], batch_size: int
) -> dict[str, Any]:
    n_segments = sum(record.n_segments for record in records)
    return {
        "training_unit": "independent_segment",
        "subject_grouped_batches": False,
        "n_segments": n_segments,
        "batch_size": batch_size,
        "n_batches": math.ceil(n_segments / batch_size),
        "coverage_ratio": 1.0,
    }


def _split_seed_sequence(config: Mapping[str, Any]) -> list[int]:
    data = config["data"]
    start = int(data["split_seed_start"])
    count = int(data["split_seed_count"])
    return list(range(start, start + count))


def _configuration_for_split_seed(
    base: PreparedConfiguration,
    split_seed: int,
    *,
    config_file: Path,
    model_source: Path,
) -> PreparedConfiguration:
    """Derive one split from inputs frozen before the multi-split run starts."""

    _positive_int(split_seed, "split_seed", allow_zero=True)
    config = copy.deepcopy(base.config)
    fractions = (
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
        float(config["data"]["test_fraction"]),
    )
    splits = stratified_subject_split(base.data_info.records, fractions, split_seed)
    description = split_manifest(splits, fractions, split_seed)
    config["resolved"]["split_seed"] = split_seed
    config["resolved"]["frozen_config_file"] = str(config_file.resolve())
    config["resolved"]["frozen_model_source_file"] = str(model_source.resolve())
    return PreparedConfiguration(
        project_root=base.project_root,
        config_file=config_file.resolve(),
        config=config,
        data_info=base.data_info,
        model_source=model_source.resolve(),
        splits=splits,
        split_description=description,
        split_seed=split_seed,
    )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _flatten_numeric_leaves(
    value: Any, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], float]:
    if isinstance(value, Mapping):
        leaves: dict[tuple[str, ...], float] = {}
        for key, item in value.items():
            leaves.update(_flatten_numeric_leaves(item, (*prefix, str(key))))
        return leaves
    if _is_finite_number(value):
        return {prefix: float(value)}
    return {}


def _insert_nested_metric(
    destination: dict[str, Any], path: tuple[str, ...], value: Mapping[str, Any]
) -> None:
    current = destination
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = dict(value)


def _summarize_values(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty numeric series")
    n_values = len(values)
    mean = sum(values) / n_values
    if n_values > 1:
        variance = sum((value - mean) ** 2 for value in values) / (n_values - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n_values,
    }


def _aggregate_numeric_metrics(
    summaries: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    collected: dict[tuple[str, ...], list[float]] = {}
    for summary in summaries:
        for path, value in _flatten_numeric_leaves(summary[field]).items():
            collected.setdefault(path, []).append(value)

    aggregated: dict[str, Any] = {}
    for path in sorted(collected):
        _insert_nested_metric(aggregated, path, _summarize_values(collected[path]))
    return aggregated


def _multi_split_summary(
    paths: ExperimentPaths,
    summaries: Sequence[Mapping[str, Any]],
    started: float,
) -> dict[str, Any]:
    if not summaries:
        raise RuntimeError("multi-split run produced no split summaries")
    warnings = [
        {"split_seed": int(summary["split_seed"]), "warning": warning}
        for summary in summaries
        for warning in summary.get("warnings", [])
    ]
    statuses = {str(summary["status"]) for summary in summaries}
    completed_statuses = {"completed", "completed_with_warning"}
    status = (
        "completed_with_warning"
        if warnings or statuses.difference({"completed"})
        else "completed"
    )
    if statuses.difference(completed_statuses):
        status = "failed"
    split_seeds = [int(summary["split_seed"]) for summary in summaries]
    return {
        "status": status,
        "completed_at": utc_now(),
        "run_directory": str(paths.root),
        "elapsed_seconds": time.monotonic() - started,
        "split_seed_start": split_seeds[0],
        "split_seed_count": len(split_seeds),
        "split_seeds": split_seeds,
        "warnings": warnings,
        "split_runs": [
            {
                "split_seed": int(summary["split_seed"]),
                "status": summary["status"],
                "run_directory": summary["run_directory"],
                "summary": str(Path(summary["run_directory"]) / "summary.json"),
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "warnings": summary.get("warnings", []),
            }
            for summary in summaries
        ],
        "final_metrics": _aggregate_numeric_metrics(summaries, "final_metrics"),
        "run_metrics": _aggregate_numeric_metrics(summaries, "run_metrics"),
        "artifacts": {
            "aggregate_final_metrics": "metrics/aggregate_final_metrics.json",
            "aggregate_run_metrics": "metrics/aggregate_run_metrics.json",
            "split_summaries": "metrics/split_summaries.json",
            "snapshot_manifest": "snapshots/manifest.json",
        },
    }


def check_configuration(
    project_root: Path,
    config_file: Path,
    run_name: str | None = None,
    device_override: str | None = None,
    split_seed_start: int | None = None,
    split_seed_count: int | None = None,
    normalize_per_channel: bool | None = None,
) -> dict[str, Any]:
    """Validate data/model/configuration without creating an experiment folder."""

    prepared = _prepare_configuration(
        project_root,
        config_file,
        run_name,
        device_override,
        split_seed_start,
        split_seed_count,
        normalize_per_channel,
    )
    model_module = load_model_module(prepared.model_source)
    edge_index, edge_weight, graph_description = build_graph(
        prepared.config["model"]["graph"],
        prepared.data_info.channel_names,
        model_module,
        prepared.project_root,
    )
    model, criterion, source_config = build_model_components(
        model_module=model_module,
        model_config=prepared.config["model"],
        loss_config=prepared.config["loss"],
        n_channels=prepared.data_info.n_channels,
        sampling_rate=prepared.data_info.sampling_rate,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    del criterion
    data_config = prepared.config["data"]
    split_summary = {
        name: {
            **{
                key: value
                for key, value in prepared.split_description["splits"][name].items()
                if key != "subjects"
            },
            "sampling": _segment_sampling_summary(
                prepared.splits[name],
                int(
                    data_config["batch_size"]
                    if name == "train"
                    else data_config["eval_batch_size"]
                ),
            ),
        }
        for name in ("train", "validation", "test")
    }
    graph_summary = {
        key: value
        for key, value in graph_description.items()
        if key not in {"edge_index", "edge_weight", "channel_names"}
    }
    return {
        "status": "valid",
        "config_file": str(prepared.config_file),
        "split_seed_start": int(prepared.config["data"]["split_seed_start"]),
        "split_seed_count": int(prepared.config["data"]["split_seed_count"]),
        "split_seeds": _split_seed_sequence(prepared.config),
        "checked_split_seed": prepared.split_seed,
        "dataset": prepared.config["resolved"],
        "splits": split_summary,
        "graph": graph_summary,
        "model": model_parameter_summary(model),
        "resolved_model_config": source_config,
        "device": str(resolve_device(str(prepared.config["experiment"]["device"]))),
    }


def _training_source_files(project_root: Path) -> list[Path]:
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    entrypoint = project_root / "train_eeg.py"
    if entrypoint.is_file():
        sources.insert(0, entrypoint)
    return sources


def _checkpoint_state(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    resolved_model_config: Mapping[str, Any],
    parameter_summary: Mapping[str, int],
    physiology_weight: float,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "epoch": epoch,
        "saved_at": utc_now(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "physiology_weight": physiology_weight,
        "selection": dict(selection),
        "model_config": dict(resolved_model_config),
        "parameter_summary": dict(parameter_summary),
        "class_mapping": {"HC": 0, "AD": 1},
        "training_unit": "independent_segment",
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _is_improved(
    current_primary: float,
    current_tiebreaker: float | None,
    best_primary: float | None,
    best_tiebreaker: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if best_primary is None:
        return True
    if mode == "min":
        if current_primary < best_primary - min_delta:
            return True
        primary_tied = abs(current_primary - best_primary) <= min_delta
    else:
        if current_primary > best_primary + min_delta:
            return True
        primary_tied = abs(current_primary - best_primary) <= min_delta
    return (
        primary_tied
        and current_tiebreaker is not None
        and best_tiebreaker is not None
        and current_tiebreaker < best_tiebreaker - 1.0e-12
    )


def _history_row(
    epoch: int,
    learning_rate: float,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    best_epoch: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
    }
    for prefix, metrics in (("train", train_metrics), ("validation", validation_metrics)):
        for name in sorted(metrics):
            row[f"{prefix}_{name}"] = metrics[name]
    return row


def _physiology_weight(epoch: int, schedule: Mapping[str, Any]) -> float:
    warmup = int(schedule["classification_warmup_epochs"])
    ramp = int(schedule["physiology_ramp_epochs"])
    target = float(schedule["target_physiology_weight"])
    if epoch <= warmup:
        return 0.0
    return target * min((epoch - warmup) / ramp, 1.0)


def _configured_loader(
    records: Sequence[SubjectRecord],
    data_config: Mapping[str, Any],
    training: bool,
    seed: int,
    pin_memory: bool,
) -> tuple[IndependentSegmentDataset, Any]:
    return make_segment_loader(
        records=records,
        training=training,
        seed=seed,
        batch_size=int(
            data_config["batch_size"]
            if training
            else data_config["eval_batch_size"]
        ),
        num_workers=int(data_config["num_workers"]),
        pin_memory=pin_memory,
    )


def _append_coverage(
    path: Path, split: str, epoch: int, report: Mapping[str, Any]
) -> None:
    entry = {"split": split, "epoch": epoch, **dict(report)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _single_class_prediction(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    return len({int(row[field]) for row in rows}) == 1


def _metric_text(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _result_groups(result: EpochResult) -> dict[str, tuple[list[dict[str, Any]], str]]:
    return {
        "segments": (result.segment_predictions, "n_segments"),
        "subject_majority_vote": (
            result.subject_majority_predictions,
            "n_subjects",
        ),
        "subject_logit_mean": (
            result.subject_logit_mean_predictions,
            "n_subjects",
        ),
    }


def _fixed_metric_groups(result: EpochResult) -> dict[str, Mapping[str, Any]]:
    return {
        "segment": result.segment_metrics,
        "subject_majority_vote": result.subject_majority_metrics,
        "subject_logit_mean": result.subject_logit_mean_metrics,
    }


def _final_prediction_artifacts(
    paths: ExperimentPaths,
    validation_result: EpochResult,
    test_result: EpochResult,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any], list[str]]:
    validation_groups = _result_groups(validation_result)
    test_groups = _result_groups(test_result)
    thresholds: dict[str, float] = {}
    validation_tuned: dict[str, Any] = {}
    test_tuned: dict[str, Any] = {}
    warnings: list[str] = []

    for family, (validation_source, count_name) in validation_groups.items():
        test_source, test_count_name = test_groups[family]
        if test_count_name != count_name:
            raise RuntimeError(f"internal count mismatch for result family {family}")
        labels = [int(row["true_label"]) for row in validation_source]
        probabilities = [float(row["probability_ad"]) for row in validation_source]
        threshold, _ = select_balanced_accuracy_threshold(labels, probabilities)
        thresholds[family] = threshold
        validation_tuned[family] = metrics_for_predictions(
            validation_source, threshold=threshold, count_name=count_name
        )
        test_tuned[family] = metrics_for_predictions(
            test_source, threshold=threshold, count_name=count_name
        )
        validation_rows = apply_decision_threshold(validation_source, threshold)
        test_rows = apply_decision_threshold(test_source, threshold)
        write_predictions(
            paths.predictions / f"validation_{family}.csv", validation_rows
        )
        write_predictions(paths.predictions / f"test_{family}.csv", test_rows)

        for split, rows in (("validation", validation_rows), ("test", test_rows)):
            if _single_class_prediction(rows, "predicted_label"):
                warnings.append(
                    f"{split} {family} fixed-threshold predictions contain one class"
                )
            if _single_class_prediction(rows, "threshold_predicted_label"):
                warnings.append(
                    f"{split} {family} tuned-threshold predictions contain one class"
                )
    return thresholds, validation_tuned, test_tuned, warnings


def _train(
    prepared: PreparedConfiguration,
    paths: ExperimentPaths,
    logger: Any,
) -> dict[str, Any]:
    config = prepared.config
    seed = int(config["experiment"]["seed"])
    deterministic = bool(config["experiment"]["deterministic"])
    seed_everything(seed, deterministic)
    device = resolve_device(str(config["experiment"]["device"]))
    logger.info("Using device %s; deterministic=%s", device, deterministic)

    model_module = load_model_module(paths.model_snapshot)
    edge_index, edge_weight, graph_description = build_graph(
        config["model"]["graph"],
        prepared.data_info.channel_names,
        model_module,
        prepared.project_root,
    )
    atomic_write_json(paths.artifacts / "channel_graph.json", graph_description)
    model, criterion, resolved_model_config = build_model_components(
        model_module=model_module,
        model_config=config["model"],
        loss_config=config["loss"],
        n_channels=prepared.data_info.n_channels,
        sampling_rate=prepared.data_info.sampling_rate,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    parameter_summary = model_parameter_summary(model)
    config["resolved"]["device"] = str(device)
    config["resolved"]["resolved_model_config"] = resolved_model_config
    config["resolved"]["parameter_summary"] = parameter_summary
    atomic_write_json(paths.root / "config" / "resolved_config.json", config)
    atomic_write_json(paths.root / "splits.json", prepared.split_description)
    logger.info(
        "Loaded %d subjects; split train/validation/test=%d/%d/%d",
        len(prepared.data_info.records),
        len(prepared.splits["train"]),
        len(prepared.splits["validation"]),
        len(prepared.splits["test"]),
    )
    logger.info(
        "Model has %d trainable parameters",
        parameter_summary["trainable_parameters"],
    )

    model = model.to(device)
    criterion = criterion.to(device)
    train_class_weight = segment_class_weights(prepared.splits["train"]).to(device)
    segment_counts = {
        diagnosis: sum(
            record.n_segments
            for record in prepared.splits["train"]
            if record.diagnosis == diagnosis
        )
        for diagnosis in ("HC", "AD")
    }
    atomic_write_json(
        paths.artifacts / "class_weights.json",
        {
            "weighting_unit": "segment",
            "segment_counts": segment_counts,
            "weights": {
                "HC": float(train_class_weight[0]),
                "AD": float(train_class_weight[1]),
            },
        },
    )
    data_config = config["data"]
    pin_memory = bool(data_config["pin_memory"] and device.type == "cuda")
    train_dataset, train_loader = _configured_loader(
        prepared.splits["train"], data_config, True, seed, pin_memory
    )
    validation_dataset, validation_loader = _configured_loader(
        prepared.splits["validation"], data_config, False, seed, pin_memory
    )
    test_dataset, test_loader = _configured_loader(
        prepared.splits["test"], data_config, False, seed, pin_memory
    )
    logger.info(
        "Per epoch: train %d independent segments in %d batches; "
        "validation %d segments in %d batches",
        len(train_dataset),
        len(train_loader),
        len(validation_dataset),
        len(validation_loader),
    )

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["min_learning_rate"]),
    )
    amp_enabled = bool(config["experiment"]["amp"] and device.type == "cuda")
    if config["experiment"]["amp"] and not amp_enabled:
        logger.info("AMP requested but disabled because the selected device is not CUDA")
    scaler = make_grad_scaler(device, amp_enabled)
    amp_dtype = str(config["experiment"]["amp_dtype"])
    max_grad_norm_value = config["training"].get("max_grad_norm")
    max_grad_norm = None if max_grad_norm_value is None else float(max_grad_norm_value)
    gradient_accumulation_steps = int(
        config["training"]["gradient_accumulation_steps"]
    )

    early = config["training"]["early_stopping"]
    monitor_name = str(early["monitor"])
    monitor_mode = str(early["mode"])
    tiebreaker_name = early.get("tiebreaker")
    min_delta = float(early["min_delta"])
    patience = int(early["patience"])
    early_start = int(early["start_epoch"])
    epochs = int(config["training"]["epochs"])
    loss_schedule = config["training"]["loss_schedule"]
    best_primary: float | None = None
    best_tiebreaker: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch = 0
    history = HistoryWriter(paths.metrics)
    coverage_path = paths.metrics / "coverage.jsonl"
    started = time.monotonic()
    atomic_write_json(
        paths.status_file,
        {
            "status": "running",
            "started_at": utc_now(),
            "pid": os.getpid(),
            "device": str(device),
            "current_epoch": 0,
            "total_epochs": epochs,
            "training_unit": "independent_segment",
        },
    )

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        current_physiology_weight = _physiology_weight(epoch, loss_schedule)
        train_dataset.set_epoch(epoch)
        train_result = run_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            device=device,
            model_module=model_module,
            class_weight=train_class_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            physiology_weight=current_physiology_weight,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        validation_dataset.set_epoch(0)
        validation_result = run_epoch(
            model=model,
            criterion=criterion,
            loader=validation_loader,
            device=device,
            model_module=model_module,
            class_weight=train_class_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            physiology_weight=current_physiology_weight,
        )
        _append_coverage(coverage_path, "train", epoch, train_result.coverage)
        _append_coverage(
            coverage_path, "validation", epoch, validation_result.coverage
        )
        scheduler.step(float(validation_result.metrics["segment_loss"]))

        monitored = validation_result.metrics.get(monitor_name)
        if monitored is None or not math.isfinite(float(monitored)):
            raise FloatingPointError(
                f"validation monitor {monitor_name!r} is unavailable/non-finite"
            )
        primary = float(monitored)
        tiebreaker = (
            None
            if tiebreaker_name is None
            else float(validation_result.metrics[str(tiebreaker_name)])
        )
        improved = _is_improved(
            primary,
            tiebreaker,
            best_primary,
            best_tiebreaker,
            monitor_mode,
            min_delta,
        )
        if improved:
            best_primary = primary
            best_tiebreaker = tiebreaker
            best_epoch = epoch
            epochs_without_improvement = 0
        elif epoch >= early_start:
            epochs_without_improvement += 1

        selection = {
            "monitor": monitor_name,
            "primary": primary,
            "tiebreaker": tiebreaker_name,
            "tiebreaker_value": tiebreaker,
        }
        checkpoint = _checkpoint_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_metrics=train_result.metrics,
            validation_metrics=validation_result.metrics,
            resolved_model_config=resolved_model_config,
            parameter_summary=parameter_summary,
            physiology_weight=current_physiology_weight,
            selection=selection,
        )
        atomic_torch_save(paths.checkpoints / "last.pt", checkpoint)
        if improved:
            atomic_torch_save(paths.checkpoints / "best.pt", checkpoint)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            _history_row(
                epoch,
                learning_rate,
                train_result.metrics,
                validation_result.metrics,
                best_epoch,
            )
        )
        atomic_write_json(
            paths.status_file,
            {
                "status": "running",
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "device": str(device),
                "current_epoch": epoch,
                "total_epochs": epochs,
                "best_epoch": best_epoch,
                "best_validation_monitor": best_primary,
                "best_validation_tiebreaker": best_tiebreaker,
                "monitor": monitor_name,
                "epochs_without_improvement": epochs_without_improvement,
                "training_unit": "independent_segment",
            },
        )
        logger.info(
            "Epoch %03d/%03d | train segments %d | train loss %.5f | "
            "val segment loss %.5f | val segment BA %s | val majority BA %s | "
            "val logit-mean BA %s | phys %.3f | lr %.3g%s",
            epoch,
            epochs,
            train_result.metrics["n_segments"],
            train_result.metrics["loss"],
            validation_result.metrics["segment_loss"],
            _metric_text(validation_result.metrics["segment_balanced_accuracy"]),
            _metric_text(
                validation_result.metrics["subject_majority_balanced_accuracy"]
            ),
            _metric_text(
                validation_result.metrics["subject_logit_mean_balanced_accuracy"]
            ),
            current_physiology_weight,
            learning_rate,
            " | best" if improved else "",
        )
        if epoch >= early_start and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d after %d non-improving eligible epochs",
                epoch,
                epochs_without_improvement,
            )
            break

    if best_epoch == 0:
        raise RuntimeError("training finished without producing a best checkpoint")
    best_checkpoint = torch.load(
        paths.checkpoints / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_physiology_weight = float(best_checkpoint["physiology_weight"])
    validation_dataset.set_epoch(0)
    final_validation = run_epoch(
        model=model,
        criterion=criterion,
        loader=validation_loader,
        device=device,
        model_module=model_module,
        class_weight=train_class_weight,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        physiology_weight=final_physiology_weight,
    )
    test_dataset.set_epoch(0)
    final_test = run_epoch(
        model=model,
        criterion=criterion,
        loader=test_loader,
        device=device,
        model_module=model_module,
        class_weight=train_class_weight,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        physiology_weight=final_physiology_weight,
    )
    _append_coverage(
        coverage_path, "validation_final", best_epoch, final_validation.coverage
    )
    _append_coverage(coverage_path, "test_final", best_epoch, final_test.coverage)

    thresholds, validation_tuned, test_tuned, warnings = _final_prediction_artifacts(
        paths, final_validation, final_test
    )
    completion_status = "completed_with_warning" if warnings else "completed"
    final_metrics = {
        "best_epoch": best_epoch,
        "monitor": monitor_name,
        "best_validation_monitor_during_training": best_primary,
        "best_validation_tiebreaker": best_tiebreaker,
        "thresholds_selected_on_validation": thresholds,
        "validation": {
            "epoch": final_validation.metrics,
            **_fixed_metric_groups(final_validation),
        },
        "validation_tuned_threshold": validation_tuned,
        "test": {
            "epoch": final_test.metrics,
            **_fixed_metric_groups(final_test),
        },
        "test_tuned_threshold": test_tuned,
    }
    atomic_write_json(paths.metrics / "final_metrics.json", final_metrics)
    elapsed = time.monotonic() - started
    summary = {
        "status": completion_status,
        "completed_at": utc_now(),
        "run_directory": str(paths.root),
        "split_seed": prepared.split_seed,
        "best_epoch": best_epoch,
        "epochs_completed": last_epoch,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "warnings": warnings,
        "parameter_summary": parameter_summary,
        "data": {
            "split_unit": "subject",
            "training_unit": "independent_segment",
            "n_subjects": len(prepared.data_info.records),
            "train_subjects": len(prepared.splits["train"]),
            "validation_subjects": len(prepared.splits["validation"]),
            "test_subjects": len(prepared.splits["test"]),
            "batch_size": int(data_config["batch_size"]),
            "eval_batch_size": int(data_config["eval_batch_size"]),
            "train_segments_per_epoch": train_result.metrics["n_segments"],
            "validation_segments_per_epoch": final_validation.metrics["n_segments"],
            "test_segments": final_test.metrics["n_segments"],
        },
        "final_metrics": final_metrics,
        "artifacts": {
            "best_checkpoint": "checkpoints/best.pt",
            "last_checkpoint": "checkpoints/last.pt",
            "history": "metrics/history.csv",
            "coverage": "metrics/coverage.jsonl",
            "validation_segment_predictions": "predictions/validation_segments.csv",
            "validation_subject_majority_vote": (
                "predictions/validation_subject_majority_vote.csv"
            ),
            "validation_subject_logit_mean": (
                "predictions/validation_subject_logit_mean.csv"
            ),
            "test_segment_predictions": "predictions/test_segments.csv",
            "test_subject_majority_vote": (
                "predictions/test_subject_majority_vote.csv"
            ),
            "test_subject_logit_mean": "predictions/test_subject_logit_mean.csv",
            "split_manifest": "splits.json",
            "snapshot_manifest": "snapshots/manifest.json",
        },
    }
    summary["run_metrics"] = {
        "best_epoch": best_epoch,
        "epochs_completed": last_epoch,
        "elapsed_seconds": elapsed,
        "train_subjects": summary["data"]["train_subjects"],
        "validation_subjects": summary["data"]["validation_subjects"],
        "test_subjects": summary["data"]["test_subjects"],
        "train_segments_per_epoch": summary["data"]["train_segments_per_epoch"],
        "validation_segments_per_epoch": summary["data"][
            "validation_segments_per_epoch"
        ],
        "test_segments": summary["data"]["test_segments"],
    }
    atomic_write_json(paths.root / "summary.json", summary)
    atomic_write_json(paths.status_file, summary)
    logger.info(
        "Training %s. Best epoch=%d; test logit-mean tuned BA=%s; results=%s",
        completion_status,
        best_epoch,
        _metric_text(test_tuned["subject_logit_mean"]["balanced_accuracy"]),
        paths.root,
    )
    return summary


def _select_sanity_records(records: Sequence[SubjectRecord]) -> list[SubjectRecord]:
    groups: dict[tuple[int, str], list[SubjectRecord]] = {}
    for record in records:
        groups.setdefault((record.label, record.institution), []).append(record)
    selected: list[SubjectRecord] = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda record: record.subject_id)
        selected.extend(candidates[:2])
    if len(selected) != 8:
        raise ValueError(
            "sanity overfit requires at least two train subjects in each "
            "diagnosis-by-institution stratum"
        )
    return selected


def _run_sanity_overfit(
    prepared: PreparedConfiguration,
    paths: ExperimentPaths,
    logger: Any,
) -> dict[str, Any]:
    config = prepared.config
    seed = int(config["experiment"]["seed"])
    deterministic = bool(config["experiment"]["deterministic"])
    seed_everything(seed, deterministic)
    device = resolve_device(str(config["experiment"]["device"]))
    selected = _select_sanity_records(prepared.splits["train"])
    logger.info(
        "Starting fixed-segment sanity overfit on subjects: %s",
        ", ".join(record.subject_id for record in selected),
    )

    model_module = load_model_module(paths.model_snapshot)
    edge_index, edge_weight, graph_description = build_graph(
        config["model"]["graph"],
        prepared.data_info.channel_names,
        model_module,
        prepared.project_root,
    )
    atomic_write_json(paths.artifacts / "channel_graph.json", graph_description)
    model, criterion, resolved_model_config = build_model_components(
        model_module=model_module,
        model_config=config["model"],
        loss_config=config["loss"],
        n_channels=prepared.data_info.n_channels,
        sampling_rate=prepared.data_info.sampling_rate,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    model = model.to(device)
    criterion = criterion.to(device)
    segment_limit = 64
    weights = segment_class_weights(selected, segment_limit).to(device)
    sanity_batch_size = min(64, int(config["data"]["batch_size"]))
    dataset, loader = make_segment_loader(
        records=selected,
        training=False,
        seed=seed,
        batch_size=sanity_batch_size,
        num_workers=min(2, int(config["data"]["num_workers"])),
        pin_memory=bool(config["data"]["pin_memory"] and device.type == "cuda"),
        max_segments_per_subject=segment_limit,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    amp_enabled = bool(config["experiment"]["amp"] and device.type == "cuda")
    scaler = make_grad_scaler(device, amp_enabled)
    history = HistoryWriter(paths.metrics)
    passed = False
    final_result: EpochResult | None = None
    started = time.monotonic()
    max_epochs = 50
    atomic_write_json(
        paths.status_file,
        {
            "status": "sanity_overfit_running",
            "started_at": utc_now(),
            "current_epoch": 0,
            "total_epochs": max_epochs,
            "device": str(device),
            "training_unit": "independent_segment",
        },
    )
    for epoch in range(1, max_epochs + 1):
        dataset.set_epoch(0)
        train_result = run_epoch(
            model=model,
            criterion=criterion,
            loader=loader,
            device=device,
            model_module=model_module,
            class_weight=weights,
            amp_enabled=amp_enabled,
            amp_dtype=str(config["experiment"]["amp_dtype"]),
            physiology_weight=0.0,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=1.0,
        )
        final_result = run_epoch(
            model=model,
            criterion=criterion,
            loader=loader,
            device=device,
            model_module=model_module,
            class_weight=weights,
            amp_enabled=amp_enabled,
            amp_dtype=str(config["experiment"]["amp_dtype"]),
            physiology_weight=0.0,
        )
        history.append(
            {
                "epoch": epoch,
                "train_optimization_loss": train_result.metrics["loss"],
                "evaluation_segment_loss": final_result.metrics["segment_loss"],
                "evaluation_segment_accuracy": final_result.metrics[
                    "segment_accuracy"
                ],
                "evaluation_subject_logit_mean_accuracy": final_result.metrics[
                    "subject_logit_mean_accuracy"
                ],
            }
        )
        atomic_torch_save(
            paths.checkpoints / "last.pt",
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": resolved_model_config,
                "metrics": final_result.metrics,
                "training_unit": "independent_segment",
                "sanity_overfit": True,
            },
        )
        logger.info(
            "Sanity epoch %02d/%02d | segment loss %.5f | segment accuracy %.3f",
            epoch,
            max_epochs,
            final_result.metrics["segment_loss"],
            final_result.metrics["segment_accuracy"],
        )
        if (
            float(final_result.metrics["segment_accuracy"]) == 1.0
            and float(final_result.metrics["segment_loss"]) <= 0.2
        ):
            passed = True
            break
    if final_result is None:
        raise RuntimeError("sanity overfit produced no result")
    write_predictions(
        paths.predictions / "train_segments.csv", final_result.segment_predictions
    )
    write_predictions(
        paths.predictions / "train_subject_majority_vote.csv",
        final_result.subject_majority_predictions,
    )
    write_predictions(
        paths.predictions / "train_subject_logit_mean.csv",
        final_result.subject_logit_mean_predictions,
    )
    status = "sanity_overfit_passed" if passed else "sanity_overfit_failed"
    summary = {
        "status": status,
        "completed_at": utc_now(),
        "run_directory": str(paths.root),
        "elapsed_seconds": time.monotonic() - started,
        "subjects": [record.subject_id for record in selected],
        "maximum_segments_per_subject": segment_limit,
        "n_independent_segments": len(dataset),
        "batch_size": sanity_batch_size,
        "classification_only": True,
        "acceptance": {"segment_accuracy": 1.0, "maximum_segment_loss": 0.2},
        "metrics": final_result.metrics,
        "checkpoint": "checkpoints/last.pt",
    }
    atomic_write_json(paths.root / "summary.json", summary)
    atomic_write_json(paths.status_file, summary)
    logger.info("Sanity overfit result: %s", status)
    return summary


def run_sanity_overfit(
    project_root: Path,
    config_file: Path,
    run_name: str | None = None,
    device_override: str | None = None,
    split_seed_start: int | None = None,
    split_seed_count: int | None = None,
    normalize_per_channel: bool | None = None,
) -> dict[str, Any]:
    """Try to memorize a fixed set of independent real segments."""

    prepared = _prepare_configuration(
        project_root,
        config_file,
        run_name,
        device_override,
        split_seed_start,
        split_seed_count,
        normalize_per_channel,
    )
    base_name = str(prepared.config["experiment"]["name"])
    prepared.config["experiment"]["name"] = f"{base_name}_sanity_overfit"
    paths = create_experiment(
        project_root=prepared.project_root,
        experiment_name=str(prepared.config["experiment"]["name"]),
        original_config=prepared.config_file,
        resolved_config=prepared.config,
        data_directory=prepared.data_info.data_directory,
        model_source=prepared.model_source,
        training_sources=_training_source_files(prepared.project_root),
    )
    logger = configure_logger(paths.logs / "train.log")
    try:
        return _run_sanity_overfit(prepared, paths, logger)
    except BaseException as error:
        (paths.logs / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        atomic_write_json(
            paths.status_file,
            {
                "status": "sanity_overfit_error",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "logs/error.log",
            },
        )
        logger.exception("Sanity overfit error: %s", error)
        raise


def run_experiment(
    project_root: Path,
    config_file: Path,
    run_name: str | None = None,
    device_override: str | None = None,
    split_seed_start: int | None = None,
    split_seed_count: int | None = None,
    normalize_per_channel: bool | None = None,
) -> dict[str, Any]:
    """Train one full experiment per consecutive split seed, then aggregate metrics."""

    initial_prepared = _prepare_configuration(
        project_root,
        config_file,
        run_name,
        device_override,
        split_seed_start,
        split_seed_count,
        normalize_per_channel,
    )
    split_seeds = _split_seed_sequence(initial_prepared.config)
    base_name = str(initial_prepared.config["experiment"]["name"])
    parent_config = copy.deepcopy(initial_prepared.config)
    parent_config["resolved"].pop("split_seed", None)
    parent_config["resolved"]["split_seeds"] = split_seeds
    parent_config["resolved"]["multi_split"] = True
    training_sources = _training_source_files(initial_prepared.project_root)
    if not training_sources:
        raise RuntimeError("multi-split training source list is empty")
    paths = create_experiment(
        project_root=initial_prepared.project_root,
        experiment_name=f"{base_name}_multi_split",
        original_config=initial_prepared.config_file,
        resolved_config=parent_config,
        data_directory=initial_prepared.data_info.data_directory,
        model_source=initial_prepared.model_source,
        training_sources=training_sources,
    )
    frozen_config_file = paths.root / "config" / "input_config.json"
    frozen_data_directory = paths.root / "snapshots" / "data_json"
    frozen_training_root = paths.root / "snapshots" / "training_code"
    frozen_training_sources = sorted(
        source for source in frozen_training_root.rglob("*.py") if source.is_file()
    )
    logger = configure_logger(paths.logs / "train.log")
    logger.info(
        "Created multi-split experiment directory %s for seeds %s",
        paths.root,
        ", ".join(str(seed) for seed in split_seeds),
    )
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    atomic_write_json(
        paths.status_file,
        {
            "status": "running",
            "started_at": utc_now(),
            "run_directory": str(paths.root),
            "split_seed_start": split_seeds[0],
            "split_seed_count": len(split_seeds),
            "split_seeds": split_seeds,
            "completed_splits": 0,
        },
    )

    try:
        for index, split_seed in enumerate(split_seeds, start=1):
            prepared = _configuration_for_split_seed(
                initial_prepared,
                split_seed,
                config_file=frozen_config_file,
                model_source=paths.model_snapshot,
            )
            prepared.config["experiment"]["name"] = f"{base_name}_split_seed_{split_seed}"
            child_paths = create_experiment(
                project_root=prepared.project_root,
                experiment_name=str(prepared.config["experiment"]["name"]),
                original_config=prepared.config_file,
                resolved_config=prepared.config,
                data_directory=frozen_data_directory,
                model_source=prepared.model_source,
                training_sources=frozen_training_sources,
                root=paths.root / f"split_seed_{split_seed}",
                training_source_root=frozen_training_root,
            )
            child_logger = configure_logger(child_paths.logs / "train.log")
            logger.info(
                "Starting split %d/%d with split_seed=%d at %s",
                index,
                len(split_seeds),
                split_seed,
                child_paths.root,
            )
            try:
                summary = _train(prepared, child_paths, child_logger)
            except BaseException as error:
                failure_traceback = traceback.format_exc()
                (child_paths.logs / "error.log").write_text(
                    failure_traceback, encoding="utf-8"
                )
                status = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                )
                atomic_write_json(
                    child_paths.status_file,
                    {
                        "status": status,
                        "failed_at": utc_now(),
                        "run_directory": str(child_paths.root),
                        "split_seed": split_seed,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": "logs/error.log",
                    },
                )
                child_logger.exception("Experiment %s: %s", status, error)
                raise
            summaries.append(summary)
            atomic_write_json(paths.metrics / "split_summaries.json", summaries)
            atomic_write_json(
                paths.status_file,
                {
                    "status": "running",
                    "updated_at": utc_now(),
                    "run_directory": str(paths.root),
                    "split_seed_start": split_seeds[0],
                    "split_seed_count": len(split_seeds),
                    "split_seeds": split_seeds,
                    "completed_splits": len(summaries),
                    "current_split_seed": split_seed,
                },
            )

        summary = _multi_split_summary(paths, summaries, started)
        atomic_write_json(
            paths.metrics / "aggregate_final_metrics.json", summary["final_metrics"]
        )
        atomic_write_json(
            paths.metrics / "aggregate_run_metrics.json", summary["run_metrics"]
        )
        atomic_write_json(paths.root / "summary.json", summary)
        atomic_write_json(paths.status_file, summary)
        logger.info(
            "Multi-split training %s. Completed %d split seeds; results=%s",
            summary["status"],
            len(split_seeds),
            paths.root,
        )
        return summary
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        (paths.logs / "error.log").write_text(failure_traceback, encoding="utf-8")
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        atomic_write_json(
            paths.status_file,
            {
                "status": status,
                "failed_at": utc_now(),
                "run_directory": str(paths.root),
                "split_seed_start": split_seeds[0],
                "split_seed_count": len(split_seeds),
                "completed_splits": len(summaries),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "logs/error.log",
            },
        )
        logger.exception("Multi-split experiment %s: %s", status, error)
        raise

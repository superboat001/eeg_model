"""PyTorch loop for independent EEG segments and post-hoc subject aggregation."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .data import (
    IndependentSegmentDataset,
    SubjectRecord,
    collate_independent_segments,
    seed_data_loader_worker,
)
from .metrics import binary_classification_metrics


@dataclass(frozen=True)
class EpochResult:
    """All independent-segment and post-hoc subject results from one epoch."""

    metrics: dict[str, Any]
    segment_metrics: dict[str, Any]
    subject_majority_metrics: dict[str, Any]
    subject_logit_mean_metrics: dict[str, Any]
    segment_predictions: list[dict[str, Any]]
    subject_majority_predictions: list[dict[str, Any]]
    subject_logit_mean_predictions: list[dict[str, Any]]
    coverage: dict[str, Any]


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {device}")
    return device


def make_segment_loader(
    records: Sequence[SubjectRecord],
    training: bool,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_segments_per_subject: int | None = None,
) -> tuple[IndependentSegmentDataset, DataLoader[dict[str, Any]]]:
    """Create a loader whose batches contain globally mixed independent segments."""

    dataset = IndependentSegmentDataset(
        records=records,
        training=training,
        seed=seed,
        max_segments_per_subject=max_segments_per_subject,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + (0 if training else 10_000))
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        # The dataset owns a deterministic global permutation so its coverage and
        # exact order remain auditable.  A second sampler shuffle is unnecessary.
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "collate_fn": collate_independent_segments,
        "worker_init_fn": seed_data_loader_worker,
        "generator": generator,
        # Recreate workers so dataset.set_epoch() is visible at every new epoch.
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return dataset, DataLoader(**loader_kwargs)


def _autocast_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")


def _segment_observations(
    batch: Mapping[str, Any], outputs: Mapping[str, Tensor]
) -> list[dict[str, Any]]:
    logits = outputs["segment_logits"].detach().float().cpu()
    if logits.ndim != 2 or logits.shape != (len(batch["subject_ids"]), 2):
        raise ValueError("model output segment_logits must have shape [batch, 2]")
    probabilities = torch.softmax(logits, dim=-1)
    quality_value = outputs.get("learned_quality_scores")
    quality = None if quality_value is None else quality_value.detach().float().cpu()
    if quality is not None and quality.shape != (logits.shape[0],):
        raise ValueError("learned_quality_scores must have shape [batch]")

    observations: list[dict[str, Any]] = []
    for row_index, subject_id in enumerate(batch["subject_ids"]):
        probability_ad = float(probabilities[row_index, 1])
        predicted_label = int(probability_ad >= 0.5)
        true_label = int(batch["labels"][row_index])
        observations.append(
            {
                "subject_id": str(subject_id),
                "institution": str(batch["institutions"][row_index]),
                "true_label": true_label,
                "true_diagnosis": str(batch["diagnoses"][row_index]),
                "segment_index": int(batch["segment_indices"][row_index]),
                "segments_available": int(batch["n_segments_total"][row_index]),
                "predicted_label": predicted_label,
                "predicted_diagnosis": "AD" if predicted_label == 1 else "HC",
                "correct": int(predicted_label == true_label),
                "probability_hc": float(probabilities[row_index, 0]),
                "probability_ad": probability_ad,
                "logit_hc": float(logits[row_index, 0]),
                "logit_ad": float(logits[row_index, 1]),
                "learned_quality_score": (
                    None if quality is None else float(quality[row_index])
                ),
            }
        )
    return observations


def _group_segments(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    for subject_id, segments in grouped.items():
        segments.sort(key=lambda row: int(row["segment_index"]))
        reference = segments[0]
        for segment in segments[1:]:
            if (
                int(segment["true_label"]) != int(reference["true_label"])
                or str(segment["true_diagnosis"])
                != str(reference["true_diagnosis"])
                or str(segment["institution"]) != str(reference["institution"])
                or int(segment["segments_available"])
                != int(reference["segments_available"])
            ):
                raise ValueError(
                    f"inconsistent segment metadata for subject {subject_id}"
                )
        indices = [int(segment["segment_index"]) for segment in segments]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate evaluated segment for subject {subject_id}")
    return dict(grouped)


def aggregate_subject_majority_vote(
    segment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate hard segment decisions; an exact vote tie resolves to AD."""

    grouped = _group_segments(segment_rows)
    results: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        segments = grouped[subject_id]
        reference = segments[0]
        ad_votes = sum(int(row["predicted_label"]) == 1 for row in segments)
        hc_votes = len(segments) - ad_votes
        vote_fraction_ad = ad_votes / len(segments)
        predicted_label = int(ad_votes >= hc_votes)
        true_label = int(reference["true_label"])
        results.append(
            {
                "aggregation_method": "majority_vote",
                "subject_id": subject_id,
                "institution": reference["institution"],
                "true_label": true_label,
                "true_diagnosis": reference["true_diagnosis"],
                "predicted_label": predicted_label,
                "predicted_diagnosis": "AD" if predicted_label == 1 else "HC",
                "correct": int(predicted_label == true_label),
                "probability_hc": 1.0 - vote_fraction_ad,
                # This score is the AD vote fraction and supports AUC/threshold metrics.
                "probability_ad": vote_fraction_ad,
                "hc_votes": hc_votes,
                "ad_votes": ad_votes,
                "vote_margin": abs(ad_votes - hc_votes),
                "vote_tie": int(ad_votes == hc_votes),
                "mean_segment_probability_ad": sum(
                    float(row["probability_ad"]) for row in segments
                )
                / len(segments),
                "segments_used": len(segments),
                "segments_available": int(reference["segments_available"]),
            }
        )
    return results


def aggregate_subject_logit_mean(
    segment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average segment logits equally within each subject, then apply softmax."""

    grouped = _group_segments(segment_rows)
    results: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        segments = grouped[subject_id]
        reference = segments[0]
        logit_hc = sum(float(row["logit_hc"]) for row in segments) / len(segments)
        logit_ad = sum(float(row["logit_ad"]) for row in segments) / len(segments)
        probability = torch.softmax(
            torch.tensor([logit_hc, logit_ad], dtype=torch.float64), dim=0
        )
        probability_ad = float(probability[1])
        predicted_label = int(probability_ad >= 0.5)
        true_label = int(reference["true_label"])
        mean_segment_probability = sum(
            float(row["probability_ad"]) for row in segments
        ) / len(segments)
        probability_std = math.sqrt(
            sum(
                (float(row["probability_ad"]) - mean_segment_probability) ** 2
                for row in segments
            )
            / len(segments)
        )
        results.append(
            {
                "aggregation_method": "logit_mean",
                "subject_id": subject_id,
                "institution": reference["institution"],
                "true_label": true_label,
                "true_diagnosis": reference["true_diagnosis"],
                "predicted_label": predicted_label,
                "predicted_diagnosis": "AD" if predicted_label == 1 else "HC",
                "correct": int(predicted_label == true_label),
                "probability_hc": float(probability[0]),
                "probability_ad": probability_ad,
                "logit_hc": logit_hc,
                "logit_ad": logit_ad,
                "mean_segment_probability_ad": mean_segment_probability,
                "segment_probability_std": probability_std,
                "segments_used": len(segments),
                "segments_available": int(reference["segments_available"]),
            }
        )
    return results


def metrics_for_predictions(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    count_name: str = "n_samples",
) -> dict[str, Any]:
    labels = [int(row["true_label"]) for row in rows]
    probabilities = [float(row["probability_ad"]) for row in rows]
    metrics = binary_classification_metrics(labels, probabilities, threshold=threshold)
    metrics[count_name] = metrics.pop("n_samples")
    return metrics


def apply_decision_threshold(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        label = int(float(row["probability_ad"]) >= threshold)
        row.update(
            {
                "decision_threshold": threshold,
                "threshold_predicted_label": label,
                "threshold_predicted_diagnosis": "AD" if label == 1 else "HC",
                "threshold_correct": int(label == int(row["true_label"])),
            }
        )
        enriched.append(row)
    return enriched


def _logit_cross_entropy(
    rows: Sequence[Mapping[str, Any]], class_weight: Tensor
) -> float:
    logits = torch.tensor(
        [[float(row["logit_hc"]), float(row["logit_ad"])] for row in rows],
        dtype=torch.float32,
    )
    labels = torch.tensor([int(row["true_label"]) for row in rows], dtype=torch.long)
    return float(
        F.cross_entropy(logits, labels, weight=class_weight.detach().float().cpu())
    )


def _merge_metric_group(
    destination: dict[str, Any], prefix: str, metrics: Mapping[str, Any]
) -> None:
    for name, value in metrics.items():
        destination[f"{prefix}_{name}"] = value


def run_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    model_module: Any,
    class_weight: Tensor,
    amp_enabled: bool,
    amp_dtype: str,
    physiology_weight: float = 0.0,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
    gradient_accumulation_steps: int = 1,
) -> EpochResult:
    """Train/evaluate independent segments, then aggregate only for reporting."""

    if physiology_weight < 0.0 or not math.isfinite(physiology_weight):
        raise ValueError("physiology_weight must be finite and non-negative")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    training = optimizer is not None
    model.train(training)
    loss_sums: defaultdict[str, float] = defaultdict(float)
    observations: list[dict[str, Any]] = []
    n_batches = 0
    n_optimizer_steps = 0
    n_segments = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.monotonic()
    compute_targets = getattr(model_module, "compute_neurophysiological_targets", None)
    model_config = getattr(model, "config")
    autocast_dtype = _autocast_dtype(amp_dtype)
    total_batches = len(loader)
    if total_batches == 0:
        raise RuntimeError("data loader has no batches")
    if training:
        optimizer.zero_grad(set_to_none=True)

    segment_weight = float(getattr(criterion, "segment_weight", 1.0))
    quality_weight = float(getattr(criterion, "quality_weight", 0.0))
    if not math.isfinite(segment_weight) or segment_weight < 0.0:
        raise ValueError("criterion.segment_weight must be finite and non-negative")
    if not math.isfinite(quality_weight) or quality_weight < 0.0:
        raise ValueError("criterion.quality_weight must be finite and non-negative")

    for batch_index, batch in enumerate(loader):
        # The model receives only a flat EEG tensor; no subject metadata is passed.
        eeg = batch["eeg"].to(device=device, non_blocking=True)
        labels = batch["labels"].to(device=device, non_blocking=True)
        batch_size = int(labels.shape[0])
        physiology_targets = None
        if physiology_weight != 0.0:
            if compute_targets is None:
                raise AttributeError(
                    "model source does not expose compute_neurophysiological_targets"
                )
            physiology_targets = compute_targets(
                eeg=eeg,
                sampling_rate=model_config.sampling_rate,
                bands=model_config.bands,
            )

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(eeg)
                source_losses = criterion(
                    outputs,
                    labels,
                    physiology_targets=physiology_targets,
                    class_weight=class_weight,
                )
                segment_loss = source_losses["segment_loss"]
                physiology_loss = source_losses["physiology_loss"]
                quality_loss = source_losses["quality_loss"]
                total_loss = (
                    segment_weight * segment_loss
                    + physiology_weight * physiology_loss
                    + quality_weight * quality_loss
                )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"non-finite loss encountered: {float(total_loss.detach())}"
                )
            if training:
                if scaler is None:
                    raise RuntimeError("a GradScaler is required for training")
                group_start = (
                    batch_index // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_size = min(
                    gradient_accumulation_steps, total_batches - group_start
                )
                scaler.scale(total_loss / group_size).backward()
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if should_step:
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    n_optimizer_steps += 1

        for name, loss in (
            ("loss", total_loss),
            ("optimization_segment_loss", segment_loss),
            ("physiology_loss", physiology_loss),
            ("quality_loss", quality_loss),
        ):
            loss_sums[name] += float(loss.detach()) * batch_size
        observations.extend(_segment_observations(batch, outputs))
        n_batches += 1
        n_segments += batch_size

    if n_segments == 0:
        raise RuntimeError("data loader yielded no EEG segments")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start_time

    observations.sort(key=lambda row: (str(row["subject_id"]), int(row["segment_index"])))
    majority_predictions = aggregate_subject_majority_vote(observations)
    logit_mean_predictions = aggregate_subject_logit_mean(observations)
    segment_metrics = metrics_for_predictions(
        observations, threshold=0.5, count_name="n_segments"
    )
    majority_metrics = metrics_for_predictions(
        majority_predictions, threshold=0.5, count_name="n_subjects"
    )
    logit_mean_metrics = metrics_for_predictions(
        logit_mean_predictions, threshold=0.5, count_name="n_subjects"
    )

    metrics = {name: value / n_segments for name, value in sorted(loss_sums.items())}
    metrics["segment_loss"] = _logit_cross_entropy(observations, class_weight)
    metrics["subject_logit_mean_loss"] = _logit_cross_entropy(
        logit_mean_predictions, class_weight
    )
    _merge_metric_group(metrics, "segment", segment_metrics)
    _merge_metric_group(metrics, "subject_majority", majority_metrics)
    _merge_metric_group(metrics, "subject_logit_mean", logit_mean_metrics)
    metrics.update(
        {
            "segment_weight": segment_weight,
            "physiology_weight": physiology_weight,
            "quality_weight": quality_weight,
            "n_batches": n_batches,
            "n_optimizer_steps": n_optimizer_steps,
            "n_segments": n_segments,
            "seconds": elapsed,
            "segments_per_second": n_segments / elapsed,
            "peak_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    coverage_function = getattr(loader.dataset, "coverage_report", None)
    coverage = coverage_function() if coverage_function is not None else {}
    for name in (
        "segments_available",
        "segments_used",
        "unique_segments",
        "duplicate_segments",
        "missing_segments",
        "coverage_ratio",
    ):
        if name in coverage:
            metrics[f"coverage_{name}"] = coverage[name]
    return EpochResult(
        metrics=metrics,
        segment_metrics=segment_metrics,
        subject_majority_metrics=majority_metrics,
        subject_logit_mean_metrics=logit_mean_metrics,
        segment_predictions=observations,
        subject_majority_predictions=majority_predictions,
        subject_logit_mean_predictions=logit_mean_predictions,
        coverage=coverage,
    )


def make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=enabled)

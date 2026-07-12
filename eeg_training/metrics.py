"""Dependency-light binary classification metrics for segments or subjects."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return None
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def binary_classification_metrics(
    labels: Sequence[int],
    probabilities_ad: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute metrics using the fixed class convention 0=HC and 1=AD."""

    if len(labels) != len(probabilities_ad) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("binary labels must be 0 or 1")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in probabilities_ad):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and lie in [0, 1]")

    predictions = [int(score >= threshold) for score in probabilities_ad]
    true_positive = sum(p == 1 and y == 1 for y, p in zip(labels, predictions))
    true_negative = sum(p == 0 and y == 0 for y, p in zip(labels, predictions))
    false_positive = sum(p == 1 and y == 0 for y, p in zip(labels, predictions))
    false_negative = sum(p == 0 and y == 1 for y, p in zip(labels, predictions))
    accuracy = (true_positive + true_negative) / len(labels)
    sensitivity = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    f1 = None
    if precision is not None and sensitivity is not None and precision + sensitivity > 0:
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)
    balanced_accuracy = None
    if sensitivity is not None and specificity is not None:
        balanced_accuracy = (sensitivity + specificity) / 2.0
    return {
        "n_samples": len(labels),
        "decision_threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity_ad": sensitivity,
        "specificity_hc": specificity,
        "precision_ad": precision,
        "f1_ad": f1,
        "roc_auc_ad": _roc_auc(labels, probabilities_ad),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_balanced_accuracy_threshold(
    labels: Sequence[int], probabilities_ad: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    """Choose a validation-only threshold with deterministic tie breaking."""

    if len(labels) != len(probabilities_ad) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    unique_scores = sorted(set(float(score) for score in probabilities_ad))
    candidates = {0.0, 0.5, 1.0}
    candidates.update(unique_scores)
    candidates.update(
        (left + right) / 2.0
        for left, right in zip(unique_scores, unique_scores[1:])
    )
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold in sorted(candidates):
        metrics = binary_classification_metrics(
            labels, probabilities_ad, threshold=threshold
        )
        balanced = metrics["balanced_accuracy"]
        sensitivity = metrics["sensitivity_ad"]
        specificity = metrics["specificity_hc"]
        if balanced is None or sensitivity is None or specificity is None:
            continue
        key = (
            float(balanced),
            min(float(sensitivity), float(specificity)),
            -abs(threshold - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("threshold selection requires both HC and AD labels")
    return best_threshold, best_metrics

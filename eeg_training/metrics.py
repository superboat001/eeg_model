"""Dependency-light binary classification metrics for segments or subjects."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute ROC-AUC from average ranks in O(N log N), including ties."""

    positive_count = sum(label == 1 for label in labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        score = ranked[start][0]
        while end < len(ranked) and ranked[end][0] == score:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in ranked[start:end]
        )
        start = end

    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _validate_binary_inputs(
    labels: Sequence[int], probabilities_ad: Sequence[float]
) -> None:
    if len(labels) != len(probabilities_ad) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("binary labels must be 0 or 1")
    if any(
        not math.isfinite(score) or not 0.0 <= score <= 1.0
        for score in probabilities_ad
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")


def binary_classification_metrics(
    labels: Sequence[int],
    probabilities_ad: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute metrics using the fixed class convention 0=HC and 1=AD."""

    _validate_binary_inputs(labels, probabilities_ad)
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
    """Choose a validation-only threshold with deterministic tie breaking.

    Candidate thresholds are evaluated by sweeping scores once in ascending order.
    This keeps the historical candidate set and tie-breaking rules while avoiding
    repeated metric/AUC computations for every threshold.
    """

    _validate_binary_inputs(labels, probabilities_ad)
    scores = [float(score) for score in probabilities_ad]
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    score_groups: list[tuple[float, int, int]] = []
    for score, label in ranked:
        if not score_groups or score != score_groups[-1][0]:
            score_groups.append((score, int(label == 1), int(label == 0)))
            continue
        previous_score, positives, negatives = score_groups[-1]
        score_groups[-1] = (
            previous_score,
            positives + int(label == 1),
            negatives + int(label == 0),
        )

    unique_scores = [score for score, _, _ in score_groups]
    candidates = {0.0, 0.5, 1.0}
    candidates.update(unique_scores)
    candidates.update(
        (left + right) / 2.0
        for left, right in zip(unique_scores, unique_scores[1:])
    )

    positive_total = sum(label == 1 for label in labels)
    negative_total = len(labels) - positive_total
    if positive_total == 0 or negative_total == 0:
        raise ValueError("threshold selection requires both HC and AD labels")

    true_positive = positive_total
    false_positive = negative_total
    true_negative = 0
    false_negative = 0
    group_index = 0
    best_threshold = 0.5
    best_key: tuple[float, float, float] | None = None
    for threshold in sorted(candidates):
        while (
            group_index < len(score_groups)
            and score_groups[group_index][0] < threshold
        ):
            _, positives, negatives = score_groups[group_index]
            true_positive -= positives
            false_negative += positives
            false_positive -= negatives
            true_negative += negatives
            group_index += 1
        sensitivity = true_positive / positive_total
        specificity = true_negative / negative_total
        balanced = (sensitivity + specificity) / 2.0
        key = (
            balanced,
            min(sensitivity, specificity),
            -abs(threshold - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold, binary_classification_metrics(
        labels, scores, threshold=best_threshold
    )

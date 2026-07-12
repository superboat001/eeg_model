"""Leakage-safe subject splitting and flat, independent EEG segment loading."""

from __future__ import annotations

import hashlib
import json
import math
import random
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SubjectRecord:
    """One preprocessed subject and its immutable metadata."""

    subject_id: str
    diagnosis: str
    label: int
    institution: str
    data_file: Path
    n_segments: int
    n_channels: int
    n_samples: int
    sampling_rate: float
    channel_names: tuple[str, ...]

    def split_entry(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "diagnosis": self.diagnosis,
            "label": self.label,
            "institution": self.institution,
            "data_file": self.data_file.name,
            "n_segments": self.n_segments,
        }


@dataclass(frozen=True)
class DatasetInfo:
    """Validated, model-relevant properties shared by all subjects."""

    data_directory: Path
    metadata_file: Path
    dataset_name: str
    records: tuple[SubjectRecord, ...]
    n_channels: int
    n_samples: int
    sampling_rate: float
    channel_names: tuple[str, ...]


def _safe_child(base: Path, relative_path: str, description: str) -> Path:
    base = base.resolve()
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{description} escapes data directory: {relative_path}") from error
    return candidate


def load_dataset_info(data_directory: Path, metadata_file: Path) -> DatasetInfo:
    """Read metadata and enforce a single channel/order/sampling contract.

    The model has learned channel embeddings and one fixed graph, so accepting a
    different channel order for any subject would silently corrupt training.  A
    mismatch is therefore a hard error rather than an implicit conversion.
    """

    data_directory = data_directory.resolve()
    metadata_file = metadata_file.resolve()
    if not data_directory.is_dir():
        raise FileNotFoundError(f"preprocessed data directory not found: {data_directory}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"dataset metadata JSON not found: {metadata_file}")

    try:
        raw = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid dataset metadata JSON: {metadata_file}") from error

    axis_order = raw.get("data_summary", {}).get("array_axis_order")
    if axis_order != ["segment", "channel", "time"]:
        raise ValueError(
            "dataset array_axis_order must be ['segment', 'channel', 'time'], "
            f"received {axis_order!r}"
        )
    subject_objects = raw.get("subjects")
    if not isinstance(subject_objects, list) or not subject_objects:
        raise ValueError("dataset metadata must contain a non-empty subjects list")

    records: list[SubjectRecord] = []
    seen_subjects: set[str] = set()
    for index, item in enumerate(subject_objects):
        if not isinstance(item, dict):
            raise ValueError(f"subjects[{index}] must be a JSON object")
        subject_id = str(item.get("subject_id", "")).strip()
        if not subject_id:
            raise ValueError(f"subjects[{index}] has no subject_id")
        if subject_id in seen_subjects:
            raise ValueError(f"duplicate subject_id in metadata: {subject_id}")
        seen_subjects.add(subject_id)

        labels = item.get("labels", {})
        diagnosis = str(labels.get("diagnosis", "")).strip()
        label = labels.get("diagnosis_id")
        institution = str(labels.get("institution", "")).strip()
        if diagnosis not in {"HC", "AD"} or label not in {0, 1}:
            raise ValueError(
                f"{subject_id}: expected HC/AD diagnosis with diagnosis_id 0/1"
            )
        if (diagnosis == "HC" and label != 0) or (diagnosis == "AD" and label != 1):
            raise ValueError(f"{subject_id}: diagnosis and diagnosis_id disagree")
        if not institution:
            raise ValueError(f"{subject_id}: institution is missing")

        shape = item.get("array_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(f"{subject_id}: array_shape must contain three positive ints")
        data_file_value = item.get("data_file")
        if not isinstance(data_file_value, str) or not data_file_value:
            raise ValueError(f"{subject_id}: data_file is missing")
        data_file = _safe_child(data_directory, data_file_value, "subject data_file")
        if not data_file.is_file():
            raise FileNotFoundError(f"{subject_id}: data file not found: {data_file}")

        sampling_rate = item.get("sampling_frequency_hz")
        if not isinstance(sampling_rate, (int, float)) or not math.isfinite(
            float(sampling_rate)
        ):
            raise ValueError(f"{subject_id}: invalid sampling_frequency_hz")
        channel_names_value = item.get("channel_names")
        if (
            not isinstance(channel_names_value, list)
            or len(channel_names_value) != shape[1]
            or any(not isinstance(name, str) or not name for name in channel_names_value)
        ):
            raise ValueError(f"{subject_id}: channel_names do not match channel dimension")

        records.append(
            SubjectRecord(
                subject_id=subject_id,
                diagnosis=diagnosis,
                label=int(label),
                institution=institution,
                data_file=data_file,
                n_segments=shape[0],
                n_channels=shape[1],
                n_samples=shape[2],
                sampling_rate=float(sampling_rate),
                channel_names=tuple(channel_names_value),
            )
        )

    reference = records[0]
    for record in records[1:]:
        if record.n_channels != reference.n_channels:
            raise ValueError("all subjects must have the same number of channels")
        if record.n_samples != reference.n_samples:
            raise ValueError("all subjects must have the same segment length")
        if not math.isclose(record.sampling_rate, reference.sampling_rate):
            raise ValueError("all subjects must have the same sampling frequency")
        if record.channel_names != reference.channel_names:
            raise ValueError(
                f"channel names/order for {record.subject_id} differ from the first subject"
            )

    return DatasetInfo(
        data_directory=data_directory,
        metadata_file=metadata_file,
        dataset_name=str(raw.get("dataset", data_directory.name)),
        records=tuple(records),
        n_channels=reference.n_channels,
        n_samples=reference.n_samples,
        sampling_rate=reference.sampling_rate,
        channel_names=reference.channel_names,
    )


def _allocate_split_counts(n_items: int, fractions: Sequence[float]) -> list[int]:
    raw_counts = [n_items * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw_counts]
    remainder = n_items - sum(counts)
    order = sorted(
        range(len(fractions)),
        key=lambda index: (raw_counts[index] - counts[index], fractions[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    positive = [index for index, fraction in enumerate(fractions) if fraction > 0]
    if n_items >= len(positive):
        for index in positive:
            if counts[index] != 0:
                continue
            donors = [candidate for candidate in positive if counts[candidate] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda candidate: counts[candidate])
            counts[donor] -= 1
            counts[index] += 1
    return counts


def stratified_subject_split(
    records: Sequence[SubjectRecord],
    fractions: Sequence[float],
    seed: int,
) -> Dict[str, list[SubjectRecord]]:
    """Split whole subjects, stratifying jointly by diagnosis and institution."""

    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(f"fractions must have {len(SPLIT_NAMES)} values")
    if any(not math.isfinite(value) or value < 0 for value in fractions):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("split fractions must sum to 1")
    if not records:
        raise ValueError("cannot split an empty dataset")

    strata: dict[tuple[int, str], list[SubjectRecord]] = defaultdict(list)
    for record in records:
        strata[(record.label, record.institution)].append(record)

    result: Dict[str, list[SubjectRecord]] = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(strata):
        items = sorted(strata[stratum], key=lambda record: record.subject_id)
        stratum_seed_bytes = hashlib.sha256(
            f"{seed}:{stratum[0]}:{stratum[1]}".encode("utf-8")
        ).digest()[:8]
        random.Random(int.from_bytes(stratum_seed_bytes, "big")).shuffle(items)
        counts = _allocate_split_counts(len(items), fractions)
        start = 0
        for split_name, count in zip(SPLIT_NAMES, counts):
            result[split_name].extend(items[start : start + count])
            start += count

    for split_name in SPLIT_NAMES:
        result[split_name].sort(key=lambda record: record.subject_id)
        if fractions[SPLIT_NAMES.index(split_name)] > 0 and not result[split_name]:
            raise ValueError(f"split {split_name} is empty; provide more subjects")

    all_ids = [record.subject_id for values in result.values() for record in values]
    if len(all_ids) != len(records) or len(set(all_ids)) != len(records):
        raise RuntimeError("internal split error: subjects are missing or duplicated")
    return result


def split_manifest(
    splits: Mapping[str, Sequence[SubjectRecord]],
    fractions: Sequence[float],
    seed: int,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "split_unit": "subject",
        "stratification": ["diagnosis", "institution"],
        "seed": seed,
        "requested_fractions": dict(zip(SPLIT_NAMES, fractions)),
        "splits": {},
    }
    for split_name in SPLIT_NAMES:
        records = list(splits[split_name])
        manifest["splits"][split_name] = {
            "n_subjects": len(records),
            "label_counts": dict(
                sorted(Counter(record.diagnosis for record in records).items())
            ),
            "institution_counts": dict(
                sorted(Counter(record.institution for record in records).items())
            ),
            "subjects": [record.split_entry() for record in records],
        }
    return manifest


class IndependentSegmentDataset(Dataset[Dict[str, Any]]):
    """Expose a globally mixed flat index of independent EEG segments.

    Subject metadata is returned solely for leakage auditing and post-hoc result
    aggregation.  It is never part of the tensor passed to the model.  The flat
    order is deterministically reshuffled every training epoch; evaluation uses
    one fixed globally shuffled order, so batches are not subject bags in either
    mode.
    """

    def __init__(
        self,
        records: Sequence[SubjectRecord],
        training: bool,
        seed: int,
        max_segments_per_subject: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("IndependentSegmentDataset requires at least one subject")
        if max_segments_per_subject is not None and max_segments_per_subject <= 0:
            raise ValueError("max_segments_per_subject must be positive or None")
        self.records = tuple(records)
        self.training = bool(training)
        self.seed = int(seed)
        self.max_segments_per_subject = max_segments_per_subject
        self.epoch = 0
        self._array_cache: dict[Path, np.ndarray] = {}
        self._selected_indices = tuple(
            self._indices_for_record(record) for record in self.records
        )
        cumulative = []
        running_total = 0
        for indices in self._selected_indices:
            running_total += len(indices)
            cumulative.append(running_total)
        self._cumulative_ends = tuple(cumulative)
        self._length = running_total
        self._order = np.arange(self._length, dtype=np.int64)
        self.set_epoch(0)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_array_cache"] = {}
        return state

    def _indices_for_record(self, record: SubjectRecord) -> np.ndarray:
        limit = self.max_segments_per_subject
        if limit is None or limit >= record.n_segments:
            return np.arange(record.n_segments, dtype=np.int64)
        digest = hashlib.sha256(
            f"segment-subset:{self.seed}:{record.subject_id}".encode("utf-8")
        ).digest()[:8]
        generator = np.random.default_rng(int.from_bytes(digest, "big"))
        return np.sort(generator.choice(record.n_segments, limit, replace=False))

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        phase = self.epoch if self.training else 0
        digest = hashlib.sha256(
            f"flat-segment-order:{self.seed}:{phase}".encode("utf-8")
        ).digest()[:8]
        generator = np.random.default_rng(int.from_bytes(digest, "big"))
        self._order = generator.permutation(self._length).astype(
            np.int64, copy=False
        )

    def _resolve_flat_index(self, flat_index: int) -> tuple[int, int]:
        record_index = bisect_right(self._cumulative_ends, flat_index)
        previous_end = 0 if record_index == 0 else self._cumulative_ends[record_index - 1]
        local_index = flat_index - previous_end
        segment_index = int(self._selected_indices[record_index][local_index])
        return record_index, segment_index

    def _load_array(self, record: SubjectRecord) -> np.ndarray:
        array = self._array_cache.get(record.data_file)
        if array is None:
            array = np.load(record.data_file, mmap_mode="r", allow_pickle=False)
            expected_shape = (record.n_segments, record.n_channels, record.n_samples)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{record.subject_id}: on-disk shape {array.shape} != metadata "
                    f"{expected_shape}"
                )
            self._array_cache[record.data_file] = array
        return array

    def __getitem__(self, index: int) -> Dict[str, Any]:
        flat_index = int(self._order[index])
        record_index, segment_index = self._resolve_flat_index(flat_index)
        record = self.records[record_index]
        array = self._load_array(record)
        # Copy one segment out of the read-only memory map before creating a tensor.
        segment = np.array(
            array[segment_index], dtype=np.float32, copy=True, order="C"
        )
        return {
            "eeg": torch.from_numpy(segment),
            "label": record.label,
            "subject_id": record.subject_id,
            "diagnosis": record.diagnosis,
            "institution": record.institution,
            "segment_index": segment_index,
            "n_segments_total": record.n_segments,
        }

    def coverage_report(self) -> Dict[str, Any]:
        total_available = sum(record.n_segments for record in self.records)
        per_subject: Dict[str, Any] = {}
        for record, indices in zip(self.records, self._selected_indices):
            per_subject[record.subject_id] = {
                "segments_available": record.n_segments,
                "segments_used": len(indices),
                "unique_segments": len(indices),
                "coverage_ratio": len(indices) / record.n_segments,
            }
        return {
            "mode": "independent_segments",
            "epoch": self.epoch,
            "n_subjects": len(self.records),
            "n_segments": self._length,
            "globally_shuffled": True,
            "subject_grouped_batches": False,
            "segments_available": total_available,
            "segments_used": self._length,
            "unique_segments": self._length,
            "duplicate_segments": 0,
            "missing_segments": total_available - self._length,
            "coverage_ratio": self._length / total_available,
            "assignment_sha256": hashlib.sha256(self._order.tobytes()).hexdigest(),
            "per_subject": per_subject,
        }


def collate_independent_segments(
    items: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Stack independent segments without adding a subject/bag dimension."""

    if not items:
        raise ValueError("cannot collate an empty batch")
    reference_shape = tuple(items[0]["eeg"].shape)
    if len(reference_shape) != 2:
        raise ValueError("each EEG segment must have shape [channels, time]")
    if any(tuple(item["eeg"].shape) != reference_shape for item in items):
        raise ValueError("all segments in a batch must share channel/time dimensions")
    return {
        "eeg": torch.stack([item["eeg"] for item in items], dim=0),
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.long),
        # Metadata below is deliberately kept outside the model input.
        "subject_ids": [str(item["subject_id"]) for item in items],
        "diagnoses": [str(item["diagnosis"]) for item in items],
        "institutions": [str(item["institution"]) for item in items],
        "segment_indices": [int(item["segment_index"]) for item in items],
        "n_segments_total": [int(item["n_segments_total"]) for item in items],
    }


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed libraries in a worker from PyTorch's per-worker initial seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def segment_class_weights(
    records: Sequence[SubjectRecord],
    max_segments_per_subject: int | None = None,
) -> Tensor:
    """Return inverse-frequency weights from independent segment counts."""

    counts = Counter(
        {
            label: sum(
                min(record.n_segments, max_segments_per_subject)
                if max_segments_per_subject is not None
                else record.n_segments
                for record in records
                if record.label == label
            )
            for label in (0, 1)
        }
    )
    if set(counts) != {0, 1} or any(counts[label] <= 0 for label in (0, 1)):
        raise ValueError("training split must contain HC and AD segments")
    total = sum(counts.values())
    return torch.tensor(
        [total / (2.0 * counts[0]), total / (2.0 * counts[1])],
        dtype=torch.float32,
    )

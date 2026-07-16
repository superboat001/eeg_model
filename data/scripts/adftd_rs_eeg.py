"""将 ADFTD-RS EEGLAB 数据预处理、切片并按被试保存为 .npy 文件。

默认使用数据集发布的 ``derivatives``：该版本已经过 0.5--45 Hz 带通、
A1/A2 重参考、ASR 和 ICA/ICLabel 去伪迹。之后仍按 BrainLat 脚本的统一流程执行
50 Hz 陷波、0.5--48 Hz 带通、固定窗口切片和片段内逐通道 z-score，以便两个
数据集得到相同格式的模型输入。也可用 ``--input-tree raw`` 从原始 BIDS 树开始。
FTD 组不会进入处理或输出；最终只保留 AD 和健康对照（HC）被试。
若输入含 EEGLAB ``boundary`` annotation，脚本会先在 annotation onset 处切开，
再对每个连续区间独立滤波和切片，因此不会产生跨数据断点的窗口。

示例：
    conda run -n cgz python scripts/adftd_rs_eeg.py

输出目录中的 ``dataset_description.json`` 会记录预处理参数、每位被试的标签、
数据 shape 和失败信息；训练脚本可使用
``eeg_preprocessing.append_model_record`` 追加模型信息。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np

from eeg_preprocessing import (
    filter_eeg_raw,
    segment_continuous_eeg,
    utc_now_iso,
    write_json,
    zscore_eeg_segments,
)


DATA_ROOT = Path("~/workspace/data").expanduser()
DEFAULT_DATASET_ROOT = DATA_ROOT / "raw/adftd-rs"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "eeg/adftd-rs"

# participants.tsv 使用 A/C/F；仅保留 A/C，并统一为与 BrainLat 兼容的 AD/HC。
GROUP_TO_DIAGNOSIS = {"A": "AD", "C": "HC"}
EXCLUDED_GROUPS = {"F": "FTD"}
ALL_GROUP_TO_DIAGNOSIS = {**GROUP_TO_DIAGNOSIS, **EXCLUDED_GROUPS}
DIAGNOSIS_TO_ID = {"HC": 0, "AD": 1}

# ADFTD-RS 的 BIDS sidecar 将全部记录的 PowerLineFrequency 标记为 50 Hz。
LINE_FREQUENCY_HZ = 50.0

# 数据集 README 记录所有 EEG 均采集于 AHEPA General Hospital 的同一站点。
INSTITUTION = "AHEPA"

# 与 BrainLat 脚本及训练模型 EEGModelConfig/EEGNetConfig 的默认值保持一致。
NORMALIZATION_EPS = 1.0e-6

BOUNDARY_ANNOTATION_DESCRIPTION = "boundary"


@dataclass
class SubjectInput:
    """一个输出 .npy 文件所对应的被试、临床标签和源文件。"""

    subject_id: str
    diagnosis: str
    source_group: str
    gender: str
    age_years: int | float
    mmse: int | float
    source_files: list[Path] = field(default_factory=list)

    @property
    def labels(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "diagnosis_id": DIAGNOSIS_TO_ID[self.diagnosis],
            "source_group": self.source_group,
            "institution": INSTITUTION,
            "source_subject_id": self.subject_id,
            "gender": self.gender,
            "age_years": self.age_years,
            "mmse": self.mmse,
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"ADFTD-RS BIDS 数据集根目录（默认：{DEFAULT_DATASET_ROOT}）。",
    )
    parser.add_argument(
        "--input-tree",
        choices=("derivatives", "raw"),
        default="derivatives",
        help=(
            "读取官方去伪迹 derivatives，或读取原始被试目录（默认：derivatives）。"
        ),
    )
    parser.add_argument(
        "--participants-file",
        type=Path,
        default=None,
        help="临床信息 TSV；默认使用 <dataset-root>/participants.tsv。",
    )
    parser.add_argument(
        "--task",
        default="eyesclosed",
        help="只处理该 BIDS task 的 .set 文件（默认：eyesclosed）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"预处理后 .npy 文件的目录（默认：{DEFAULT_OUTPUT_DIR}）。",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="每个切片的时长（秒，默认：1.0）。",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=0.5,
        help="相邻切片起点的间隔（秒，默认：0.5，即 50%% 重叠）。",
    )
    parser.add_argument(
        "--description-name",
        default="dataset_description.json",
        help="输出目录内描述文件的名称。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同名 .npy 文件和描述文件。",
    )
    return parser.parse_args()


def _parse_number(value: str, *, field_name: str, subject_id: str) -> int | float:
    """解析临床数值，并在能无损表示时使用整数写入 JSON。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{subject_id} 的 {field_name} 不是有效数值：{value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"{subject_id} 的 {field_name} 必须是有限数：{value!r}")
    return int(number) if number.is_integer() else number


def read_participants(participants_file: Path) -> dict[str, SubjectInput]:
    """读取并严格校验 ADFTD-RS 的 ``participants.tsv``。"""
    participants_file = participants_file.expanduser()
    if not participants_file.is_file():
        raise FileNotFoundError(f"participants.tsv 不存在：{participants_file}")

    required_columns = {"participant_id", "Gender", "Age", "Group", "MMSE"}
    subjects: dict[str, SubjectInput] = {}
    with participants_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        available_columns = set(reader.fieldnames or ())
        missing_columns = sorted(required_columns - available_columns)
        if missing_columns:
            raise ValueError(
                f"{participants_file} 缺少必需列：{', '.join(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            subject_id = (row.get("participant_id") or "").strip()
            if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject_id):
                raise ValueError(
                    f"{participants_file}:{line_number} 的 participant_id "
                    f"不符合 BIDS sub-<label> 格式：{subject_id!r}"
                )
            if subject_id in subjects:
                raise ValueError(f"participants.tsv 中被试 ID 重复：{subject_id}")

            source_group = (row.get("Group") or "").strip().upper()
            try:
                diagnosis = ALL_GROUP_TO_DIAGNOSIS[source_group]
            except KeyError as error:
                raise ValueError(
                    f"{subject_id} 的 Group 必须是 A、C 或 F，实际为："
                    f"{source_group!r}"
                ) from error

            gender = (row.get("Gender") or "").strip().upper()
            if gender not in {"F", "M"}:
                raise ValueError(
                    f"{subject_id} 的 Gender 必须是 F 或 M，实际为：{gender!r}"
                )

            age_years = _parse_number(
                row.get("Age") or "",
                field_name="Age",
                subject_id=subject_id,
            )
            mmse = _parse_number(
                row.get("MMSE") or "",
                field_name="MMSE",
                subject_id=subject_id,
            )
            if age_years <= 0:
                raise ValueError(f"{subject_id} 的 Age 必须大于 0：{age_years}")
            if not 0 <= mmse <= 30:
                raise ValueError(f"{subject_id} 的 MMSE 必须在 [0, 30]：{mmse}")

            subjects[subject_id] = SubjectInput(
                subject_id=subject_id,
                diagnosis=diagnosis,
                source_group=source_group,
                gender=gender,
                age_years=age_years,
                mmse=mmse,
            )

    if not subjects:
        raise ValueError(f"participants.tsv 不含被试记录：{participants_file}")
    return subjects


def discover_subjects(
    eeg_root: Path,
    participants_file: Path,
    *,
    task: str,
) -> list[SubjectInput]:
    """关联 BIDS ``sub-*`` 目录中的 .set 文件与临床标签。"""
    eeg_root = eeg_root.expanduser()
    if not eeg_root.is_dir():
        raise FileNotFoundError(f"EEG 输入目录不存在：{eeg_root}")
    if not task or not re.fullmatch(r"[A-Za-z0-9]+", task):
        raise ValueError(f"task 必须是非空 BIDS 标签，实际为：{task!r}")

    subjects = read_participants(participants_file)
    files_by_subject: dict[str, list[Path]] = defaultdict(list)
    task_pattern = re.compile(r"(?:^|_)task-([^_]+)(?:_|$)")

    # 只进入 eeg_root 直属的 BIDS sub-* 树。input_tree=raw 时 eeg_root 也是
    # 数据集根目录，这可避免把其下的 derivatives 再扫描一遍。
    for eeg_file in sorted(eeg_root.glob("sub-*/**/*.set")):
        task_match = task_pattern.search(eeg_file.name)
        if task_match is None or task_match.group(1) != task:
            continue

        relative_path = eeg_file.relative_to(eeg_root)
        if len(relative_path.parts) < 3:
            raise ValueError(
                "无法从路径解析 BIDS 被试；预期至少为 "
                f"<sub-ID>/eeg/<文件>.set，实际为：{eeg_file}"
            )
        subject_id = relative_path.parts[0]
        if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject_id):
            raise ValueError(
                f"EEG 根目录应直接包含 sub-* 目录，无法解析：{eeg_file}"
            )
        if "eeg" not in relative_path.parts[1:-1]:
            raise ValueError(f".set 文件不在 BIDS eeg 目录中：{eeg_file}")
        if not eeg_file.name.startswith(f"{subject_id}_"):
            raise ValueError(
                f"目录被试 ID {subject_id} 与文件名不一致：{eeg_file.name}"
            )
        files_by_subject[subject_id].append(eeg_file)

    metadata_only = sorted(set(subjects) - set(files_by_subject))
    eeg_only = sorted(set(files_by_subject) - set(subjects))
    if metadata_only or eeg_only:
        problems = []
        if metadata_only:
            problems.append(
                f"participants.tsv 中有 {len(metadata_only)} 名被试缺少 task-{task} EEG："
                + ", ".join(metadata_only[:10])
                + (" ..." if len(metadata_only) > 10 else "")
            )
        if eeg_only:
            problems.append(
                f"有 {len(eeg_only)} 个 EEG 被试缺少临床记录："
                + ", ".join(eeg_only[:10])
                + (" ..." if len(eeg_only) > 10 else "")
            )
        raise ValueError("\n".join(problems))

    for subject_id, source_files in files_by_subject.items():
        subjects[subject_id].source_files.extend(source_files)
    return sorted(
        (
            subject
            for subject in subjects.values()
            if subject.diagnosis in DIAGNOSIS_TO_ID
        ),
        key=lambda item: item.subject_id,
    )


def ensure_output_is_safe(
    subjects: list[SubjectInput],
    output_dir: Path,
    description_path: Path,
    *,
    overwrite: bool,
) -> None:
    """在写入前检查目标文件，避免无意覆盖已有预处理结果。"""
    if overwrite:
        return

    existing = [
        output_dir / f"{subject.subject_id}.npy"
        for subject in subjects
        if (output_dir / f"{subject.subject_id}.npy").exists()
    ]
    if description_path.exists():
        existing.append(description_path)
    if existing:
        preview = "\n".join(f"  - {path}" for path in existing[:10])
        suffix = "\n  ..." if len(existing) > 10 else ""
        raise FileExistsError(
            "检测到已有输出，为避免覆盖已停止。若确认需要重建，请添加 --overwrite：\n"
            f"{preview}{suffix}"
        )


def split_continuous_blocks(
    raw: Any,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """按 EEGLAB ``boundary`` onset 返回左闭右开的连续采样区间。

    ``boundary`` 的 duration 表示从原记录中删除的数据时长，不对应当前数组中
    仍需跳过的一段样本。因此只使用 onset 作为切点。重复切点会合并，位于记录
    首尾的切点会记录但不会生成空区间。
    """
    boundary_annotations = [
        (float(onset), float(duration), str(description))
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
        if str(description).strip().casefold()
        == BOUNDARY_ANNOTATION_DESCRIPTION.casefold()
    ]
    if not boundary_annotations:
        return [(0, int(raw.n_times))], []

    onsets = [annotation[0] for annotation in boundary_annotations]
    if raw.annotations.orig_time is None:
        # 无绝对时间参考时，annotation onset 包含 first_time 偏移；
        # time_as_index(origin=None) 则要求相对于 first_samp 的秒数。
        times = [onset - float(raw.first_time) for onset in onsets]
        origin = None
    else:
        times = onsets
        origin = raw.annotations.orig_time
    sample_indices = raw.time_as_index(
        times,
        use_rounding=True,
        origin=origin,
    )

    split_samples: set[int] = set()
    annotation_records: list[dict[str, Any]] = []
    for (onset, duration, description), sample_index_value in zip(
        boundary_annotations,
        sample_indices,
    ):
        sample_index = int(sample_index_value)
        if not 0 < sample_index < raw.n_times:
            status = "ignored_at_recording_edge"
        elif sample_index in split_samples:
            status = "duplicate_split_point_ignored"
        else:
            split_samples.add(sample_index)
            status = "used_as_split_point"
        annotation_records.append(
            {
                "description": description,
                "onset_seconds": onset,
                "duration_seconds": duration,
                "split_sample_index": sample_index,
                "status": status,
            }
        )

    boundaries = [0, *sorted(split_samples), int(raw.n_times)]
    blocks = [
        (start, stop)
        for start, stop in zip(boundaries, boundaries[1:])
        if stop > start
    ]
    if not blocks:
        raise ValueError("boundary 切分后没有非空连续 EEG 区间。")
    return blocks, annotation_records


def process_subject(
    subject: SubjectInput,
    *,
    window_seconds: float,
    step_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """读取记录，按 boundary 分区后独立滤波、切片并合并。"""
    session_fragments: list[np.ndarray] = []
    session_records: list[dict[str, Any]] = []
    reference_sfreq: float | None = None
    reference_channels: list[str] | None = None
    filter_description: dict[str, Any] | None = None

    for eeg_file in subject.source_files:
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose="ERROR")
        eeg_picks = mne.pick_types(
            raw.info,
            meg=False,
            eeg=True,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            exclude=[],
        )
        if len(eeg_picks) == 0:
            raise ValueError(f"文件不含 MNE 标记为 EEG 的通道：{eeg_file}")

        sampling_frequency = float(raw.info["sfreq"])
        channel_names = [raw.ch_names[index] for index in eeg_picks]
        data = raw.get_data(picks=eeg_picks)

        if reference_sfreq is None:
            reference_sfreq = sampling_frequency
            reference_channels = channel_names
        elif sampling_frequency != reference_sfreq or channel_names != reference_channels:
            raise ValueError(
                f"被试 {subject.subject_id} 的多个 .set 文件采样率或 EEG 通道不一致，"
                "不能合并为同一个 .npy 文件。"
            )

        if window_seconds <= 0 or step_seconds <= 0:
            raise ValueError("window_seconds 和 step_seconds 都必须大于 0。")
        window_samples = round(window_seconds * sampling_frequency)
        if window_samples <= 0:
            raise ValueError("窗口时长换算后的采样点数必须大于 0。")

        continuous_blocks, boundary_annotations = split_continuous_blocks(raw)
        eeg_info = mne.pick_info(
            raw.info,
            [int(index) for index in eeg_picks],
            copy=True,
        )
        block_records: list[dict[str, Any]] = []
        session_segment_count = 0
        for block_index, (start, stop) in enumerate(continuous_blocks):
            n_samples = stop - start
            block_record: dict[str, Any] = {
                "block_index": block_index,
                "start_sample_inclusive": start,
                "stop_sample_exclusive": stop,
                "start_seconds": start / sampling_frequency,
                "stop_seconds_exclusive": stop / sampling_frequency,
                "n_samples": n_samples,
            }
            if n_samples < window_samples:
                block_record.update(
                    {
                        "status": "dropped_too_short_for_one_complete_window",
                        "n_segments": 0,
                    }
                )
                block_records.append(block_record)
                continue

            # RawArray 只包含当前连续区间；滤波器不可能读取相邻区间的数据。
            block_raw = mne.io.RawArray(
                data[:, start:stop],
                eeg_info.copy(),
                verbose="ERROR",
            )
            current_filter_description = filter_eeg_raw(
                block_raw,
                l_freq=0.5,
                h_freq=48.0,
                line_frequency=LINE_FREQUENCY_HZ,
                picks="eeg",
            )
            if filter_description is None:
                filter_description = current_filter_description

            fragments, segment_description = segment_continuous_eeg(
                block_raw.get_data(),
                sampling_frequency,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
            )
            fragments = zscore_eeg_segments(fragments, eps=NORMALIZATION_EPS)
            session_fragments.append(fragments)
            session_segment_count += int(fragments.shape[0])
            block_record.update(
                {
                    "status": "processed",
                    "n_segments": segment_description["n_segments"],
                }
            )
            block_records.append(block_record)

        used_split_samples = sorted(
            {
                int(annotation["split_sample_index"])
                for annotation in boundary_annotations
                if annotation["status"] == "used_as_split_point"
            }
        )
        dropped_blocks = [
            block
            for block in block_records
            if block["status"] == "dropped_too_short_for_one_complete_window"
        ]
        session_records.append(
            {
                "source_file": str(eeg_file),
                "input_shape": [int(data.shape[0]), int(data.shape[1])],
                "n_segments": session_segment_count,
                "boundary_handling": {
                    "annotation_description": BOUNDARY_ANNOTATION_DESCRIPTION,
                    "policy": "split_at_annotation_onset_before_filtering",
                    "boundary_duration_interpretation": (
                        "metadata_for_removed_source_data_not_samples_to_drop_from_"
                        "the_current_array"
                    ),
                    "boundary_annotations": boundary_annotations,
                    "n_boundary_annotations": len(boundary_annotations),
                    "split_sample_indices": used_split_samples,
                    "n_continuous_blocks": len(continuous_blocks),
                    "n_processed_blocks": len(block_records) - len(dropped_blocks),
                    "n_dropped_short_blocks": len(dropped_blocks),
                    "n_dropped_short_block_samples": sum(
                        int(block["n_samples"]) for block in dropped_blocks
                    ),
                },
                "continuous_block_records": block_records,
            }
        )

    if not session_fragments or filter_description is None or reference_sfreq is None:
        raise RuntimeError(f"未找到可处理的 .set 文件：{subject.subject_id}")

    merged_fragments = np.concatenate(session_fragments, axis=0)
    subject_record = {
        "subject_id": subject.subject_id,
        "labels": subject.labels,
        "source_files": [str(path) for path in subject.source_files],
        "session_records": session_records,
        "data_file": f"{subject.subject_id}.npy",
        "array_shape": [int(value) for value in merged_fragments.shape],
        "dtype": str(merged_fragments.dtype),
        "sampling_frequency_hz": reference_sfreq,
        "channel_names": reference_channels,
        "filtering": filter_description,
    }
    return merged_fragments, subject_record


def build_description(
    *,
    arguments: argparse.Namespace,
    eeg_root: Path,
    participants_file: Path,
    processed_subjects: list[dict[str, Any]],
    failed_subjects: list[dict[str, str]],
    n_discovered_subjects: int,
) -> dict[str, Any]:
    """构建随数据一起保存的、供训练使用的描述文件内容。"""
    shapes = {
        subject["subject_id"]: subject["array_shape"]
        for subject in processed_subjects
    }
    diagnosis_counts = Counter(
        subject["labels"]["diagnosis"] for subject in processed_subjects
    )
    source_group_counts = Counter(
        subject["labels"]["source_group"] for subject in processed_subjects
    )
    gender_counts = Counter(
        subject["labels"]["gender"] for subject in processed_subjects
    )
    institution_counts = Counter(
        subject["labels"]["institution"] for subject in processed_subjects
    )
    total_segments = sum(subject["array_shape"][0] for subject in processed_subjects)

    segmentation = {
        "method": "fixed_length_sliding_window",
        "window_seconds": arguments.window_seconds,
        "step_seconds": arguments.step_seconds,
        "overlap_seconds": max(arguments.window_seconds - arguments.step_seconds, 0.0),
        "tail_handling": "drop_samples_that_cannot_form_a_complete_window",
        "boundary_handling": (
            "segment_each_continuous_block_independently_and_never_cross_boundaries"
        ),
        "output_axis_order": ["segment", "channel", "time"],
        "sampling_rate_policy": "read_from_each_EEGLAB_file",
    }
    preprocessing = {
        "channel_selection": "only MNE channels typed as EEG",
        "operation_order": [
            "split_at_boundary_annotations",
            "notch_filter",
            "bandpass_filter",
            "segmentation",
            "zscore_normalization",
        ],
        "boundary_handling": {
            "annotation_description_case_insensitive": (
                BOUNDARY_ANNOTATION_DESCRIPTION
            ),
            "split_location": "annotation_onset_rounded_to_nearest_sample",
            "continuous_interval_convention": "left_closed_right_open",
            "filtering_scope": "each_continuous_block_independently",
            "segmentation_scope": "each_continuous_block_independently",
            "cross_boundary_windows": "forbidden",
            "short_block_handling": "drop_if_shorter_than_one_complete_window",
            "annotation_duration_interpretation": (
                "duration_of_data_removed_from_the_source_recording; do_not_remove_"
                "that_duration_again_from_the_current_array"
            ),
        },
        "notch_filter": {
            "frequency_hz": LINE_FREQUENCY_HZ,
            "selection_basis": (
                "ADFTD-RS 的全部原始 BIDS EEG sidecar 均将 PowerLineFrequency "
                "标记为 50 Hz，采集地点位于希腊。"
            ),
            "method": "fir",
            "phase": "zero",
            "fir_design": "firwin",
        },
        "bandpass_filter": {
            "highpass_hz": 0.5,
            "lowpass_hz": 48.0,
            "method": "fir",
            "phase": "zero",
            "fir_design": "firwin",
        },
        "zscore_normalization": {
            "method": "amplitude_scaled_regularized_zscore",
            "scope": "each channel within each segment independently",
            "axis": "time",
            "amplitude_scale": "maximum_absolute_value_along_time_axis",
            "zero_amplitude_scale_handling": "replace_with_one",
            "standard_deviation_ddof": 0,
            "epsilon": NORMALIZATION_EPS,
            "channel_mask": "all_saved_EEG_channels_are_valid",
            "constant_channel_handling": "output_all_zeros",
        },
    }

    input_preprocessing = (
        {
            "stage": "published_derivatives",
            "description": (
                "ADFTD-RS 发布者提供的去伪迹数据：0.5--45 Hz Butterworth 带通、"
                "A1/A2 重参考、ASR、RunICA，并通过 ICLabel 去除眼动和下颌伪迹成分。"
            ),
        }
        if arguments.input_tree == "derivatives"
        else {
            "stage": "published_raw",
            "description": "ADFTD-RS 原始 BIDS .set 数据，未使用发布者的 ASR/ICA derivatives。",
        }
    )

    return {
        "schema_version": "1.0",
        "dataset": "adftd-rs",
        "created_at": utc_now_iso(),
        "software": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "numpy": np.__version__,
        },
        "output_directory": str(arguments.output_dir.expanduser()),
        "source": {
            "format": "BIDS EEGLAB .set",
            "dataset_root": str(arguments.dataset_root.expanduser()),
            "input_tree": arguments.input_tree,
            "eeg_root": str(eeg_root),
            "participants_file": str(participants_file),
            "task": arguments.task,
            "subject_id_rule": "BIDS participant_id (sub-<label>)",
            "multiple_set_files_per_subject": "concatenate_segments_along_axis_0",
            "input_preprocessing": input_preprocessing,
        },
        "preprocessing": preprocessing,
        "segmentation": segmentation,
        "labels": {
            "definitions": {
                "diagnosis": {
                    "mapping": DIAGNOSIS_TO_ID,
                    "source_group_mapping": GROUP_TO_DIAGNOSIS,
                    "excluded_source_groups": EXCLUDED_GROUPS,
                    "description": (
                        "HC=0，AD=1；participants.tsv 中的健康组 C 统一命名为 HC，"
                        "FTD 组 F 被明确排除。"
                    ),
                },
                "institution": {
                    "description": (
                        "所有记录均采集于 AHEPA General Hospital，使用固定值 AHEPA"
                    ),
                },
                "gender": {
                    "description": "participants.tsv 的 Gender：F=female，M=male",
                },
                "age_years": {
                    "description": "participants.tsv 的 Age，单位为年",
                },
                "mmse": {
                    "description": "Mini-Mental State Examination，范围 0--30",
                },
            },
            "counts": {
                "diagnosis": dict(sorted(diagnosis_counts.items())),
                "source_group": dict(sorted(source_group_counts.items())),
                "institution": dict(sorted(institution_counts.items())),
                "gender": dict(sorted(gender_counts.items())),
            },
        },
        "data_summary": {
            "storage": "one_npy_file_per_subject",
            "array_axis_order": ["segment", "channel", "time"],
            "number_of_discovered_subjects": n_discovered_subjects,
            "number_of_saved_subjects": len(processed_subjects),
            "number_of_failed_subjects": len(failed_subjects),
            "total_segments": total_segments,
            "array_shape_per_subject": shapes,
        },
        "subjects": processed_subjects,
        "failed_subjects": failed_subjects,
        "model_runs": [],
    }


def main() -> None:
    arguments = parse_arguments()
    arguments.dataset_root = arguments.dataset_root.expanduser()
    arguments.output_dir = arguments.output_dir.expanduser()
    participants_file = (
        arguments.participants_file.expanduser()
        if arguments.participants_file is not None
        else arguments.dataset_root / "participants.tsv"
    )
    eeg_root = (
        arguments.dataset_root / "derivatives"
        if arguments.input_tree == "derivatives"
        else arguments.dataset_root
    )
    description_path = arguments.output_dir / arguments.description_name

    subjects = discover_subjects(
        eeg_root,
        participants_file,
        task=arguments.task,
    )
    ensure_output_is_safe(
        subjects,
        arguments.output_dir,
        description_path,
        overwrite=arguments.overwrite,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    processed_subjects: list[dict[str, Any]] = []
    failed_subjects: list[dict[str, str]] = []
    for index, subject in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] 处理 {subject.subject_id}")
        try:
            fragments, subject_record = process_subject(
                subject,
                window_seconds=arguments.window_seconds,
                step_seconds=arguments.step_seconds,
            )
            np.save(
                arguments.output_dir / subject_record["data_file"],
                fragments,
                allow_pickle=False,
            )
            processed_subjects.append(subject_record)
        except Exception as error:  # 记录单个被试问题后继续处理其余被试。
            failed_subjects.append(
                {
                    "subject_id": subject.subject_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            print(f"[FAILED] {subject.subject_id}: {type(error).__name__}: {error}")

    description = build_description(
        arguments=arguments,
        eeg_root=eeg_root,
        participants_file=participants_file,
        processed_subjects=processed_subjects,
        failed_subjects=failed_subjects,
        n_discovered_subjects=len(subjects),
    )
    write_json(description_path, description)
    print(
        f"完成：保存 {len(processed_subjects)}/{len(subjects)} 名被试；"
        f"描述文件：{description_path}"
    )


if __name__ == "__main__":
    main()

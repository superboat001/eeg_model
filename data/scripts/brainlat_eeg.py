"""将 BrainLat EEGLAB 数据预处理、切片并按被试保存为 .npy 文件。

示例（只在确认输入目录和工频后执行）：
    conda run -n cgz python brainlat_eeg.py

BrainLat 的 AR（阿根廷）与 CL（智利）站点均使用 50 Hz 市电，因此本脚本将工频
陷波固定为 50 Hz。输出目录中的 ``dataset_description.json`` 会记录本次预处理的
参数、数据 shape、每位被试标签及标签统计；训练脚本可使用
``eeg_preprocessing.append_model_record`` 追加模型信息。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np

from eeg_preprocessing import (
    filter_eeg_raw,
    segment_continuous_eeg,
    utc_now_iso,
    write_json,
)


DATA_ROOT = Path("~/workspace/data").expanduser()
DEFAULT_AD_ROOT = DATA_ROOT / "raw/brainlat/1_AD"
DEFAULT_HC_ROOT = DATA_ROOT / "raw/brainlat/5_HC"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "eeg/brainlat"
DIAGNOSIS_TO_ID = {"HC": 0, "AD": 1}

# BrainLat 数据仅包含 AR（Argentina）和 CL（Chile）两个采集站点，均为 50 Hz 市电。
# 该值不提供命令行覆盖，以防在同一数据集上混用不同的工频设置。
LINE_FREQUENCY_HZ = 50.0


@dataclass
class SubjectInput:
    """一个输出 .npy 文件所对应的一名被试及其所有 .set 源文件。"""

    subject_id: str
    diagnosis: str
    institution: str
    source_subject_id: str
    source_files: list[Path] = field(default_factory=list)

    @property
    def labels(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "diagnosis_id": DIAGNOSIS_TO_ID[self.diagnosis],
            "institution": self.institution,
            "source_subject_id": self.source_subject_id,
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ad-root",
        type=Path,
        default=DEFAULT_AD_ROOT,
        help=f"AD 原始数据目录（默认：{DEFAULT_AD_ROOT}）",
    )
    parser.add_argument(
        "--hc-root",
        type=Path,
        default=DEFAULT_HC_ROOT,
        help=f"HC 原始数据目录（默认：{DEFAULT_HC_ROOT}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"预处理后 .npy 文件的目录（默认：{DEFAULT_OUTPUT_DIR}）",
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


def _safe_identifier(*parts: str) -> str:
    """用诊断、机构和原始被试 ID 组成稳定且可作文件名的唯一 ID。"""
    safe_parts = []
    for part in parts:
        safe_part = re.sub(r"[^A-Za-z0-9._-]+", "-", part.strip()).strip("-.")
        if not safe_part:
            raise ValueError(f"不能从 {part!r} 生成安全的被试标识。")
        safe_parts.append(safe_part)
    return "__".join(safe_parts)


def discover_subjects(
    dataset_roots: Iterable[tuple[str, Path]],
) -> list[SubjectInput]:
    """按 ``诊断/机构/原始被试 ID`` 收集 .set 文件。

    BrainLat 的预期布局为 ``<类别根目录>/<机构>/<被试>/<可选子目录>/*.set``。
    同一被试下多个 .set 文件会切片后沿 segment 维度合并到一个 .npy 文件。
    """
    subjects: dict[str, SubjectInput] = {}
    for diagnosis, root in dataset_roots:
        root = root.expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"{diagnosis} 原始数据目录不存在：{root}")

        for eeg_file in sorted(root.rglob("*.set")):
            relative_path = eeg_file.relative_to(root)
            if len(relative_path.parts) < 3:
                raise ValueError(
                    "无法从路径解析机构和被试 ID；预期至少为 "
                    f"<机构>/<被试>/<文件>.set，实际为：{eeg_file}"
                )
            institution, source_subject_id = relative_path.parts[:2]
            subject_id = _safe_identifier(
                diagnosis,
                institution,
                source_subject_id,
            )
            subject = subjects.setdefault(
                subject_id,
                SubjectInput(
                    subject_id=subject_id,
                    diagnosis=diagnosis,
                    institution=institution,
                    source_subject_id=source_subject_id,
                ),
            )
            subject.source_files.append(eeg_file)

    return sorted(subjects.values(), key=lambda item: item.subject_id)


def ensure_output_is_safe(
    subjects: Iterable[SubjectInput],
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


def process_subject(
    subject: SubjectInput,
    *,
    window_seconds: float,
    step_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """读取一名被试的所有会话，滤波、切片并在 segment 维度拼接。"""
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

        filter_description = filter_eeg_raw(
            raw,
            l_freq=0.5,
            h_freq=48.0,
            line_frequency=LINE_FREQUENCY_HZ,
            picks=eeg_picks,
        )
        sampling_frequency = float(raw.info["sfreq"])
        channel_names = [raw.ch_names[index] for index in eeg_picks]
        data = raw.get_data(picks=eeg_picks)
        fragments, segment_description = segment_continuous_eeg(
            data,
            sampling_frequency,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )

        if reference_sfreq is None:
            reference_sfreq = sampling_frequency
            reference_channels = channel_names
        elif sampling_frequency != reference_sfreq or channel_names != reference_channels:
            raise ValueError(
                f"被试 {subject.subject_id} 的多个 .set 文件采样率或 EEG 通道不一致，"
                "不能合并为同一个 .npy 文件。"
            )

        session_fragments.append(fragments)
        session_records.append(
            {
                "source_file": str(eeg_file),
                "input_shape": segment_description["input_shape"],
                "n_segments": segment_description["n_segments"],
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
        "output_axis_order": ["segment", "channel", "time"],
        "sampling_rate_policy": "read_from_each_EEGLAB_file",
    }
    preprocessing = {
        "channel_selection": "only MNE channels typed as EEG",
        "operation_order": ["notch_filter", "bandpass_filter"],
        "notch_filter": {
            "frequency_hz": LINE_FREQUENCY_HZ,
            "selection_basis": (
                "BrainLat 的 AR（Argentina）和 CL（Chile）采集站点均使用 50 Hz 市电；"
                "对四个站点/类别组合抽样的 11 份可读取记录未见稳定的 50 或 60 Hz 谱峰，"
                "且原始 .set 元数据表明发布版本已做 0.5–40 Hz 过滤。"
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
    }

    return {
        "schema_version": "1.0",
        "dataset": "brainlat",
        "created_at": utc_now_iso(),
        "software": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "numpy": np.__version__,
        },
        "output_directory": str(arguments.output_dir.expanduser()),
        "source": {
            "format": "EEGLAB .set",
            "ad_root": str(arguments.ad_root.expanduser()),
            "hc_root": str(arguments.hc_root.expanduser()),
            "subject_id_rule": "{diagnosis}__{institution}__{source_subject_id}",
            "multiple_set_files_per_subject": "concatenate_segments_along_axis_0",
        },
        "preprocessing": preprocessing,
        "segmentation": segmentation,
        "labels": {
            "definitions": {
                "diagnosis": {
                    "mapping": DIAGNOSIS_TO_ID,
                    "description": "HC=0，AD=1",
                },
                "institution": {
                    "description": "从原始目录的第一层子目录读取",
                },
            },
            "counts": {
                "diagnosis": dict(sorted(diagnosis_counts.items())),
                "institution": dict(sorted(institution_counts.items())),
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
    arguments.ad_root = arguments.ad_root.expanduser()
    arguments.hc_root = arguments.hc_root.expanduser()
    arguments.output_dir = arguments.output_dir.expanduser()
    description_path = arguments.output_dir / arguments.description_name

    subjects = discover_subjects(
        (("AD", arguments.ad_root), ("HC", arguments.hc_root))
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

"""可复用的 EEG 预处理、切片和描述文件工具。

数据集专有脚本负责定位数据、定义标签和组织输出；本文件只放可被多个
数据集复用的处理步骤。训练脚本也可以使用 ``append_model_record`` 向同一份
描述文件追加模型信息。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def utc_now_iso() -> str:
    """返回带时区的 UTC 时间，便于写入可追溯的描述文件。"""
    return datetime.now(timezone.utc).isoformat()


def _validate_filter_arguments(
    l_freq: float | None,
    h_freq: float | None,
    line_frequency: float | None,
) -> None:
    if l_freq is not None and l_freq <= 0:
        raise ValueError("l_freq 必须大于 0 或为 None。")
    if h_freq is not None and h_freq <= 0:
        raise ValueError("h_freq 必须大于 0 或为 None。")
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError("l_freq 必须小于 h_freq。")
    if line_frequency is not None and line_frequency not in (50.0, 60.0):
        raise ValueError("工频陷波频率只能是 50 Hz 或 60 Hz。")


def filter_eeg_raw(
    raw: Any,
    *,
    l_freq: float | None,
    h_freq: float | None,
    line_frequency: float | None,
    picks: Sequence[int] | str = "eeg",
    verbose: bool | str | None = "ERROR",
) -> dict[str, Any]:
    """原地对 MNE ``Raw`` 对象进行 EEG 滤波。

    陷波先于带通滤波执行。即使低通截止频率低于 50/60 Hz，这个顺序也能明确
    消除工频成分，并会在描述文件中保留实际的处理顺序。
    """
    _validate_filter_arguments(l_freq, h_freq, line_frequency)

    sampling_frequency = float(raw.info["sfreq"])
    nyquist = sampling_frequency / 2
    if h_freq is not None and h_freq >= nyquist:
        raise ValueError(
            f"低通截止频率 {h_freq} Hz 必须小于 Nyquist 频率 {nyquist} Hz。"
        )
    if line_frequency is not None and line_frequency >= nyquist:
        raise ValueError(
            f"陷波频率 {line_frequency} Hz 必须小于 Nyquist 频率 {nyquist} Hz。"
        )

    operations: list[dict[str, Any]] = []
    if line_frequency is not None:
        raw.notch_filter(
            freqs=[line_frequency],
            picks=picks,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=verbose,
        )
        operations.append(
            {
                "name": "notch_filter",
                "parameters": {
                    "frequencies_hz": [line_frequency],
                    "method": "fir",
                    "phase": "zero",
                    "fir_design": "firwin",
                },
            }
        )

    if l_freq is not None or h_freq is not None:
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            picks=picks,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=verbose,
        )
        operations.append(
            {
                "name": "bandpass_filter",
                "parameters": {
                    "l_freq_hz": l_freq,
                    "h_freq_hz": h_freq,
                    "method": "fir",
                    "phase": "zero",
                    "fir_design": "firwin",
                },
            }
        )

    return {
        "applied_to": "MNE channels typed as EEG",
        "sampling_frequency_hz": sampling_frequency,
        "operations": operations,
    }


def segment_continuous_eeg(
    data: np.ndarray,
    sampling_frequency: float,
    *,
    window_seconds: float = 1.0,
    step_seconds: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """将连续 EEG 切为固定长度、可重叠的片段。

    返回数组形状为 ``(n_segments, n_channels, n_samples_per_segment)``。末尾无法
    构成完整窗口的样本会被丢弃；采样率由原始数据决定，而不是写死为 512 Hz。
    """
    if data.ndim != 2:
        raise ValueError(f"期望二维连续 EEG 数据，实际 shape 为 {data.shape}。")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency 必须大于 0。")
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds 和 step_seconds 都必须大于 0。")

    window_samples = round(window_seconds * sampling_frequency)
    step_samples = round(step_seconds * sampling_frequency)
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("窗口长度和步长换算后的采样点数必须大于 0。")

    n_channels, n_samples = data.shape
    if n_samples < window_samples:
        raise ValueError(
            f"数据仅有 {n_samples} 个采样点，少于一个 {window_samples} 点窗口。"
        )

    starts = range(0, n_samples - window_samples + 1, step_samples)
    fragments = np.stack(
        [data[:, start : start + window_samples] for start in starts], axis=0
    )
    description = {
        "method": "fixed_length_sliding_window",
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "overlap_seconds": max(window_seconds - step_seconds, 0.0),
        "sampling_frequency_hz": float(sampling_frequency),
        "window_samples": window_samples,
        "step_samples": step_samples,
        "tail_handling": "drop_samples_that_cannot_form_a_complete_window",
        "output_axis_order": ["segment", "channel", "time"],
        "n_segments": int(fragments.shape[0]),
        "input_shape": [int(n_channels), int(n_samples)],
        "output_shape": [int(value) for value in fragments.shape],
    }
    return fragments, description


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化 {type(value).__name__} 到 JSON。")


def write_json(path: Path, content: Mapping[str, Any]) -> None:
    """原子写入 UTF-8 JSON，避免训练或预处理意外留下半份描述文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            default=_json_default,
        )
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def append_model_record(
    description_path: Path,
    model_information: Mapping[str, Any],
) -> None:
    """向数据描述文件追加一次训练的模型信息。

    训练脚本可传入例如模型名称、超参数、训练/验证/测试被试划分、随机种子、
    指标和模型权重路径。调用方不需要手工维护 ``model_runs`` 列表。
    """
    if not description_path.is_file():
        raise FileNotFoundError(f"描述文件不存在：{description_path}")

    with description_path.open("r", encoding="utf-8") as file:
        description = json.load(file)

    model_record = dict(model_information)
    model_record.setdefault("recorded_at", utc_now_iso())
    description.setdefault("model_runs", []).append(model_record)
    write_json(description_path, description)

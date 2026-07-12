"""Dynamic model loading and reproducible EEG graph construction."""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def load_model_module(source_file: Path) -> ModuleType:
    """Load one model source file without writing bytecode beside that source."""

    source_file = source_file.resolve()
    if not source_file.is_file():
        raise FileNotFoundError(f"model source file not found: {source_file}")
    module_name = f"experiment_model_{abs(hash(str(source_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None:
        raise ImportError(f"cannot create import spec for model: {source_file}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses consult sys.modules while decorating classes.
    import sys

    sys.modules[module_name] = module
    try:
        # Executing a SourceFileLoader normally creates ``__pycache__`` next to
        # the configured model.  The model directory is an immutable input to
        # this framework, so compile the same source directly instead.
        code = compile(source_file.read_bytes(), str(source_file), "exec")
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _standard_montage_knn(
    channel_names: Sequence[str], montage_name: str, neighbors: int
) -> tuple[Tensor, None, dict[str, Any]]:
    if neighbors < 1 or neighbors >= len(channel_names):
        raise ValueError("graph.neighbors must lie in [1, n_channels)")
    try:
        import mne
    except ImportError as error:
        raise ImportError(
            "standard_montage_knn graph construction requires the mne package"
        ) from error

    montage = mne.channels.make_standard_montage(montage_name)
    positions = montage.get_positions()["ch_pos"]
    missing = [name for name in channel_names if name not in positions]
    if missing:
        raise ValueError(
            f"montage {montage_name!r} has no positions for channels: {missing[:8]}"
        )
    coordinates = np.asarray([positions[name] for name in channel_names], dtype=np.float64)
    if coordinates.shape != (len(channel_names), 3) or not np.isfinite(coordinates).all():
        raise ValueError("montage returned invalid 3D channel coordinates")
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    edges: set[tuple[int, int]] = set()
    for source in range(len(channel_names)):
        nearest = np.argsort(distances[source])[1 : neighbors + 1]
        for target_value in nearest:
            target = int(target_value)
            edges.add(tuple(sorted((source, target))))
    ordered_edges = sorted(edges)
    edge_index = torch.tensor(ordered_edges, dtype=torch.long).t().contiguous()
    metadata = {
        "type": "standard_montage_knn",
        "montage": montage_name,
        "neighbors_requested_per_channel": neighbors,
        "n_undirected_edges": len(ordered_edges),
        "weighting": "uniform",
    }
    return edge_index, None, metadata


def _json_graph(
    graph_file: Path, n_channels: int
) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
    try:
        raw = json.loads(graph_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid graph JSON: {graph_file}") from error
    edge_index = torch.as_tensor(raw.get("edge_index"), dtype=torch.long)
    if edge_index.ndim != 2:
        raise ValueError("graph JSON edge_index must be a two-dimensional array")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t()
    if edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
        raise ValueError("graph JSON edge_index must have shape [2, E] or [E, 2]")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= n_channels:
        raise ValueError("graph JSON contains a channel index outside [0, n_channels)")
    edge_weight_value = raw.get("edge_weight")
    edge_weight = None
    if edge_weight_value is not None:
        edge_weight = torch.as_tensor(edge_weight_value, dtype=torch.float32)
        if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.shape[1]:
            raise ValueError("graph JSON edge_weight must have shape [E]")
        if not torch.isfinite(edge_weight).all() or (edge_weight < 0).any():
            raise ValueError("graph JSON edge weights must be finite and non-negative")
    metadata = {
        "type": "json",
        "source_file": str(graph_file),
        "n_input_edges": edge_index.shape[1],
        "weighted": edge_weight is not None,
    }
    return edge_index.contiguous(), edge_weight, metadata


def build_graph(
    graph_config: Mapping[str, Any],
    channel_names: Sequence[str],
    model_module: ModuleType,
    project_root: Path,
) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
    """Build the model graph without using validation/test signals."""

    graph_type = str(graph_config.get("type", "")).strip()
    n_channels = len(channel_names)
    if graph_type == "standard_montage_knn":
        edge_index, edge_weight, metadata = _standard_montage_knn(
            channel_names,
            str(graph_config.get("montage", "biosemi128")),
            int(graph_config.get("neighbors", 4)),
        )
    elif graph_type == "ring":
        hops = int(graph_config.get("hops", 1))
        helper = getattr(model_module, "make_ring_edge_index", None)
        if helper is None:
            raise AttributeError("model source does not expose make_ring_edge_index")
        edge_index = helper(n_channels, hops=hops)
        edge_weight = None
        metadata = {
            "type": "ring",
            "hops": hops,
            "n_undirected_edges": int(edge_index.shape[1]),
            "warning": "ring is a baseline/debug graph, not a montage-derived graph",
        }
    elif graph_type == "json":
        source_value = graph_config.get("source_file")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError("graph.type=json requires graph.source_file")
        graph_file = Path(source_value)
        if not graph_file.is_absolute():
            graph_file = project_root / graph_file
        if not graph_file.is_file():
            raise FileNotFoundError(f"graph JSON not found: {graph_file}")
        edge_index, edge_weight, metadata = _json_graph(
            graph_file.resolve(), n_channels
        )
    else:
        raise ValueError(
            "graph.type must be one of: standard_montage_knn, ring, json"
        )

    metadata.update(
        {
            "n_channels": n_channels,
            "channel_names": list(channel_names),
            "edge_index": edge_index.t().tolist(),
            "edge_weight": None if edge_weight is None else edge_weight.tolist(),
        }
    )
    return edge_index, edge_weight, metadata


def build_model_components(
    model_module: ModuleType,
    model_config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    n_channels: int,
    sampling_rate: float,
    edge_index: Tensor,
    edge_weight: Tensor | None,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Instantiate the configured model and its source-provided multi-task loss."""

    config_class_name = str(model_config.get("config_class", "EEGModelConfig"))
    model_class_name = str(model_config.get("class_name", "EEGSegmentClassifier"))
    loss_class_name = str(model_config.get("loss_class", "EEGMultiTaskLoss"))
    config_class = getattr(model_module, config_class_name, None)
    model_class = getattr(model_module, model_class_name, None)
    loss_class = getattr(model_module, loss_class_name, None)
    if config_class is None or model_class is None or loss_class is None:
        raise AttributeError(
            "model source is missing one of the configured model/config/loss classes"
        )

    parameters = dict(model_config.get("parameters", {}))
    if "n_channels" in parameters or "sampling_rate" in parameters:
        raise ValueError(
            "model.parameters must not set n_channels/sampling_rate; they are inferred"
        )
    if "bands" in parameters:
        parameters["bands"] = tuple(tuple(band) for band in parameters["bands"])
    if "temporal_kernels" in parameters:
        parameters["temporal_kernels"] = tuple(parameters["temporal_kernels"])
    source_config = config_class(
        n_channels=n_channels,
        sampling_rate=sampling_rate,
        **parameters,
    )
    model = model_class(source_config, edge_index=edge_index, edge_weight=edge_weight)
    criterion = loss_class(**dict(loss_config))
    resolved_source_config = asdict(source_config)
    return model, criterion, resolved_source_config


def model_parameter_summary(model: nn.Module) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "buffer_elements": int(buffers),
    }

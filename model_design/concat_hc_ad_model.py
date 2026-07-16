"""Segment-level HC/AD classification for independently sampled EEG segments.

This module contains model definitions and training-facing helpers only.  It does
not contain a data loader, an optimizer, or a training loop.  The main input is a
flat batch of segments with shape ``[segments, channels, samples]``.  Every
segment is encoded, trained, and predicted independently; no subject identity or
cross-segment aggregation is used.  Class order is fixed to ``0 = HC`` and
``1 = AD``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Band = Tuple[str, float, float]


DEFAULT_BANDS: Tuple[Band, ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class EEGModelConfig:
    """Configuration for :class:`EEGSegmentClassifier`."""

    n_channels: int
    sampling_rate: float
    bands: Tuple[Band, ...] = DEFAULT_BANDS
    fir_kernel_size: int = 129
    d_model: int = 48
    segment_dim: int = 96
    ssm_state_dim: int = 24
    temporal_kernels: Tuple[int, ...] = (7, 15, 31)
    temporal_stride: int = 4
    local_kernel_size: int = 5
    local_dilation: int = 2
    attention_hidden: int = 32
    quality_hidden: int = 64
    dropout: float = 0.10
    num_classes: int = 2
    normalize_per_channel: bool = True
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.n_channels < 2:
            raise ValueError("n_channels must be at least 2")
        if not math.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be finite and positive")
        if self.fir_kernel_size < 3 or self.fir_kernel_size % 2 == 0:
            raise ValueError("fir_kernel_size must be an odd integer >= 3")
        if self.d_model <= 0 or self.segment_dim <= 0 or self.ssm_state_dim <= 0:
            raise ValueError("model and state dimensions must be positive")
        if not self.temporal_kernels:
            raise ValueError("temporal_kernels cannot be empty")
        if any(k < 3 or k % 2 == 0 for k in self.temporal_kernels):
            raise ValueError("all temporal kernels must be odd integers >= 3")
        if self.d_model % len(self.temporal_kernels) != 0:
            raise ValueError("d_model must be divisible by len(temporal_kernels)")
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be >= 1")
        if self.local_kernel_size < 3 or self.local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be an odd integer >= 3")
        if self.local_dilation < 1:
            raise ValueError("local_dilation must be >= 1")
        if self.attention_hidden <= 0 or self.quality_hidden <= 0:
            raise ValueError("attention_hidden and quality_hidden must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.eps) or not 0.0 < self.eps < 0.5:
            raise ValueError("eps must be finite and lie strictly between 0 and 0.5")
        if self.num_classes != 2:
            raise ValueError("this HC/AD model requires num_classes=2")
        if not isinstance(self.normalize_per_channel, bool):
            raise ValueError("normalize_per_channel must be boolean")
        _validate_bands(self.bands, self.sampling_rate)
        band_names = {name for name, _, _ in self.bands}
        if not {"theta", "alpha"}.issubset(band_names):
            raise ValueError(
                "bands must include named theta and alpha entries for the auxiliary ratio"
            )


def _validate_bands(bands: Sequence[Band], sampling_rate: float) -> None:
    if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be finite and positive")
    if not bands:
        raise ValueError("at least one frequency band is required")
    nyquist = sampling_rate / 2.0
    previous_high = -math.inf
    names = set()
    for name, low, high in bands:
        if name in names:
            raise ValueError(f"duplicate band name: {name}")
        names.add(name)
        if not (0.0 <= low < high < nyquist):
            raise ValueError(
                f"invalid band {name}=({low}, {high}) for Nyquist={nyquist}; "
                "the upper edge must be strictly below Nyquist"
            )
        if low < previous_high:
            raise ValueError("bands must be ordered and non-overlapping")
        previous_high = high


def make_ring_edge_index(n_channels: int, hops: int = 1) -> Tensor:
    """Create one copy of each undirected edge in a ring graph.

    This helper is intended for smoke tests and examples, not as a substitute for
    a montage-derived anatomical graph.
    """

    if n_channels < 2:
        raise ValueError("n_channels must be at least 2")
    if not 1 <= hops < n_channels:
        raise ValueError("hops must be in [1, n_channels)")
    edges = set()
    for source in range(n_channels):
        for hop in range(1, hops + 1):
            target = (source + hop) % n_channels
            a, b = sorted((source, target))
            if a != b:
                edges.add((a, b))
    ordered = sorted(edges)
    return torch.tensor(ordered, dtype=torch.long).t().contiguous()


def _prepare_undirected_graph(
    n_channels: int,
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Remove self loops, then symmetrize/deduplicate a nonnegative graph."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.numel() == 0:
        raise ValueError("edge_index cannot be empty")
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    if int(edges.min()) < 0 or int(edges.max()) >= n_channels:
        raise ValueError("edge_index contains a channel index outside [0, C)")

    n_edges = edges.shape[1]
    if edge_weight is None:
        weights = torch.ones(n_edges, dtype=torch.float32)
    else:
        if edge_weight.ndim != 1 or edge_weight.numel() != n_edges:
            raise ValueError("edge_weight must have shape [E]")
        weights = edge_weight.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(weights).all():
            raise ValueError("edge_weight must be finite")
        if (weights < 0).any():
            raise ValueError("only nonnegative graph weights are supported")

    non_self = edges[0] != edges[1]
    edges = edges[:, non_self]
    weights = weights[non_self]
    if edges.shape[1] == 0:
        raise ValueError("the graph must contain at least one non-self edge")
    reverse = edges.flip(0)
    all_edges = torch.cat((edges, reverse), dim=1)
    all_weights = torch.cat((weights, weights), dim=0)

    # Average duplicates.  This accepts either one edge per undirected pair or an
    # already bidirectional edge list without accidentally doubling its strength.
    keys = all_edges[0] * n_channels + all_edges[1]
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    weight_sum = torch.zeros(unique_keys.numel(), dtype=torch.float32)
    counts = torch.zeros_like(weight_sum)
    weight_sum.index_add_(0, inverse, all_weights)
    counts.index_add_(0, inverse, torch.ones_like(all_weights))
    unique_weights = weight_sum / counts.clamp_min(1.0)
    source = torch.div(unique_keys, n_channels, rounding_mode="floor")
    target = unique_keys.remainder(n_channels)
    unique_edges = torch.stack((source, target), dim=0)
    return unique_edges, unique_weights


def _masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Softmax with exact zeros outside the mask, including all-masked rows."""

    if mask.dtype != torch.bool:
        mask = mask.bool()
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )


def _normalization_groups(n_channels: int) -> int:
    for groups in (8, 4, 2):
        if n_channels % groups == 0:
            return groups
    return 1


class FixedBandpassFilterBank(nn.Module):
    """Non-trainable windowed-sinc FIR decomposition."""

    def __init__(
        self,
        sampling_rate: float,
        bands: Sequence[Band],
        kernel_size: int,
    ) -> None:
        super().__init__()
        _validate_bands(bands, sampling_rate)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.sampling_rate = float(sampling_rate)
        self.band_names = tuple(band[0] for band in bands)
        self.kernel_size = int(kernel_size)
        filters = self._design_filters(bands)
        self.register_buffer("fir_filters", filters, persistent=True)

    def _design_filters(self, bands: Sequence[Band]) -> Tensor:
        n = torch.arange(self.kernel_size, dtype=torch.float64)
        n = n - (self.kernel_size - 1) / 2.0
        window = torch.hamming_window(
            self.kernel_size, periodic=False, dtype=torch.float64
        )
        filters = []
        for _, low, high in bands:
            lowpass_high = (2.0 * high / self.sampling_rate) * torch.sinc(
                2.0 * high * n / self.sampling_rate
            )
            lowpass_low = (2.0 * low / self.sampling_rate) * torch.sinc(
                2.0 * low * n / self.sampling_rate
            )
            kernel = (lowpass_high - lowpass_low) * window
            kernel = kernel - kernel.mean()  # suppress residual DC leakage

            center_frequency = (low + high) / 2.0
            phase = 2.0 * math.pi * center_frequency * n / self.sampling_rate
            real = torch.sum(kernel * torch.cos(phase))
            imag = -torch.sum(kernel * torch.sin(phase))
            gain = torch.sqrt(real.square() + imag.square()).clamp_min(1.0e-12)
            filters.append((kernel / gain).to(torch.float32))
        return torch.stack(filters, dim=0).unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        """Return fixed-band signals with shape ``[N, C, F, T]``."""

        if x.ndim != 3:
            raise ValueError("filter-bank input must have shape [N, C, T]")
        n_segments, n_channels, n_samples = x.shape
        if n_samples < 2:
            raise ValueError("EEG segments must contain at least two samples")
        pad = self.kernel_size // 2
        flat = x.reshape(n_segments * n_channels, 1, n_samples)
        original_dtype = flat.dtype
        padding_mode = "reflect" if n_samples > pad else "replicate"

        # Keep the fixed FIR numerically stable under mixed precision.  The result
        # is cast back so downstream trainable layers can still use autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat_float = flat.float()
            padded = F.pad(flat_float, (pad, pad), mode=padding_mode)
            filtered = F.conv1d(padded, self.fir_filters.float())
        filtered = filtered.to(dtype=original_dtype)
        return filtered.reshape(
            n_segments, n_channels, len(self.band_names), n_samples
        )


class MultiScaleTemporalStem(nn.Module):
    """Shared per-channel multi-scale temporal convolution stem."""

    def __init__(
        self,
        d_model: int,
        kernels: Sequence[int],
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % len(kernels) != 0:
            raise ValueError("d_model must be divisible by number of kernels")
        branch_dim = d_model // len(kernels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        1,
                        branch_dim,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(_normalization_groups(branch_dim), branch_dim),
                    nn.SiLU(),
                )
                for kernel in kernels
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(_normalization_groups(d_model), d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``[..., T]`` into ``[..., L, D]`` without mixing nodes."""

        if x.ndim < 2:
            raise ValueError("temporal stem input must end in a sample dimension")
        leading_shape = x.shape[:-1]
        flat = x.reshape(-1, 1, x.shape[-1])
        branches = [branch(flat) for branch in self.branches]
        lengths = {branch.shape[-1] for branch in branches}
        if len(lengths) != 1:
            raise RuntimeError("multi-scale branches produced inconsistent lengths")
        encoded = self.mix(torch.cat(branches, dim=1)).transpose(1, 2)
        return encoded.reshape(*leading_shape, encoded.shape[-2], encoded.shape[-1])


class BandAttentionEncoder(nn.Module):
    """Encode each fixed band and aggregate bands at every node/time window."""

    def __init__(
        self,
        n_bands: int,
        d_model: int,
        kernels: Sequence[int],
        stride: int,
        dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.stride = stride
        self.eps = eps
        self.stem = MultiScaleTemporalStem(d_model, kernels, stride, dropout)
        self.band_embedding = nn.Parameter(torch.empty(n_bands, d_model))
        nn.init.normal_(self.band_embedding, mean=0.0, std=0.02)
        self.energy_projection = nn.Linear(1, d_model, bias=False)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(8, d_model // 2)),
            nn.Tanh(),
            nn.Linear(max(8, d_model // 2), 1),
        )
        self.energy_score = nn.Linear(1, 1, bias=False)

    def forward(self, bands: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return aggregate, attention, and centered local log-energy.

        Args:
            bands: Tensor with shape ``[N, C, F, T]``.
        """

        if bands.ndim != 4 or bands.shape[2] != self.n_bands:
            raise ValueError("bands must have shape [N, C, configured_bands, T]")
        encoded = self.stem(bands)  # [N, C, F, L, D]
        n, c, f, n_windows, _ = encoded.shape

        squared = bands.float().square().reshape(n * c * f, 1, bands.shape[-1])
        local_power = F.avg_pool1d(
            squared,
            kernel_size=self.stride,
            stride=self.stride,
            ceil_mode=True,
        )
        if local_power.shape[-1] != n_windows:
            local_power = F.interpolate(
                local_power, size=n_windows, mode="linear", align_corners=False
            )
        log_energy = torch.log(local_power.clamp_min(self.eps))
        log_energy = log_energy.reshape(n, c, f, n_windows, 1)
        centered_log_energy = log_energy - log_energy.mean(dim=2, keepdim=True)
        centered_log_energy = centered_log_energy.clamp(-12.0, 12.0)
        centered_log_energy = centered_log_energy.to(dtype=encoded.dtype)

        encoded = encoded + self.band_embedding[None, None, :, None, :]
        encoded = encoded + self.energy_projection(centered_log_energy)
        scores = self.score(encoded) + self.energy_score(centered_log_energy)
        attention = torch.softmax(scores, dim=2)
        aggregate = torch.sum(attention * encoded, dim=2)
        return aggregate, attention.squeeze(-1), centered_log_energy.squeeze(-1)


class AdaptiveFeatureGate(nn.Module):
    """Scalar gate per channel and time window for two aligned features."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        hidden = max(8, d_model // 2)
        self.gate = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        if a.shape != b.shape:
            raise ValueError("features passed to a gate must have identical shapes")
        a_norm = self.norm_a(a)
        b_norm = self.norm_b(b)
        evidence = torch.cat(
            (a_norm, b_norm, torch.abs(a_norm - b_norm), a_norm * b_norm), dim=-1
        )
        gate = self.gate(evidence)
        return gate * a + (1.0 - gate) * b, gate


class LocalTemporalConv(nn.Module):
    """Normal and dilated depthwise temporal convolutions with a residual path."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.local = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.dilated = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            groups=d_model,
            bias=False,
        )
        self.mix = nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("local temporal block expects [N, C, L, D]")
        n, c, length, d_model = x.shape
        normalized = self.input_norm(x).reshape(n * c, length, d_model)
        normalized = normalized.transpose(1, 2)
        features = torch.cat((self.local(normalized), self.dilated(normalized)), dim=1)
        update = F.glu(self.mix(features), dim=1).transpose(1, 2)
        update = update.reshape(n, c, length, d_model)
        return self.output_norm(x + self.dropout(update))


class SelectiveDiagonalSSM(nn.Module):
    """A small input-selective, stable diagonal state-space recurrence."""

    def __init__(self, d_model: int, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.delta_projection = nn.Linear(d_model, state_dim)
        self.candidate_projection = nn.Linear(d_model, state_dim)
        self.output_projection = nn.Linear(state_dim, d_model, bias=False)
        self.output_gate = nn.Linear(d_model, d_model)
        self.log_rate = nn.Parameter(torch.empty(state_dim))
        self.skip = nn.Parameter(torch.ones(d_model))
        nn.init.constant_(self.delta_projection.bias, -2.0)
        nn.init.uniform_(self.log_rate, -2.0, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Process ``[M, L, D]`` in its current sequence direction."""

        if x.ndim != 3:
            raise ValueError("SSM expects [sequences, time, features]")
        delta = F.softplus(self.delta_projection(x)) + 1.0e-4
        candidate = torch.tanh(self.candidate_projection(x))
        rate = F.softplus(self.log_rate).to(dtype=x.dtype) + 1.0e-4
        state = x.new_zeros(x.shape[0], self.state_dim)
        outputs = []
        for time_index in range(x.shape[1]):
            decay = torch.exp(-delta[:, time_index] * rate)
            state = decay * state + (1.0 - decay) * candidate[:, time_index]
            outputs.append(self.output_projection(state))
        state_output = torch.stack(outputs, dim=1)
        gate = torch.sigmoid(self.output_gate(x))
        return gate * state_output + self.skip * x


class BidirectionalLightSSM(nn.Module):
    """Independent forward/backward SSMs followed by residual fusion."""

    def __init__(self, d_model: int, state_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.forward_ssm = SelectiveDiagonalSSM(d_model, state_dim)
        self.backward_ssm = SelectiveDiagonalSSM(d_model, state_dim)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("bidirectional SSM expects [N, C, L, D]")
        n, c, length, d_model = x.shape
        sequence = self.input_norm(x).reshape(n * c, length, d_model)
        forward = self.forward_ssm(sequence)
        backward_input = torch.flip(sequence, dims=(1,))
        backward = torch.flip(self.backward_ssm(backward_input), dims=(1,))
        merged = self.merge(torch.cat((forward, backward), dim=-1))
        merged = merged.reshape(n, c, length, d_model)
        return self.output_norm(x + self.dropout(merged))


class SparseGraphConv(nn.Module):
    """One sparse message-passing layer over channels at each time window.

    Graph normalization is recomputed per segment when a channel mask is supplied,
    so a missing node neither sends nor receives messages.
    """

    def __init__(
        self,
        n_channels: int,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        d_model: int,
        dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        edges, weights = _prepare_undirected_graph(
            n_channels, edge_index, edge_weight
        )
        self.n_channels = n_channels
        self.eps = eps
        self.register_buffer("edge_index", edges, persistent=True)
        self.register_buffer("edge_weight", weights, persistent=True)
        self.neighbor_projection = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def aggregate(self, x: Tensor, channel_mask: Tensor) -> Tuple[Tensor, Tensor]:
        n, c, length, d_model = x.shape
        if c != self.n_channels:
            raise ValueError(
                f"graph was built for {self.n_channels} channels, received {c}"
            )
        if channel_mask.shape != (n, c):
            raise ValueError("channel_mask passed to graph must have shape [N, C]")

        source, target = self.edge_index
        base_weight = self.edge_weight.unsqueeze(0)
        active = channel_mask[:, source] & channel_mask[:, target]
        active_weight = base_weight * active.to(dtype=base_weight.dtype)

        # Symmetric D^-1/2 A D^-1/2 normalization, dynamically masked per segment.
        degree = torch.zeros(n, c, device=x.device, dtype=torch.float32)
        degree.scatter_add_(
            1, target.unsqueeze(0).expand(n, -1), active_weight.float()
        )
        denominator = torch.sqrt(
            degree[:, source].clamp_min(self.eps)
            * degree[:, target].clamp_min(self.eps)
        )
        normalized_weight = (active_weight.float() / denominator).to(dtype=x.dtype)

        messages = x[:, source] * normalized_weight[:, :, None, None]
        flat_target = (
            torch.arange(n, device=x.device)[:, None] * c + target[None, :]
        ).reshape(-1)
        aggregate = x.new_zeros(n * c, length, d_model)
        aggregate.index_add_(0, flat_target, messages.reshape(-1, length, d_model))
        has_neighbor = degree > 0.0
        return aggregate.reshape(n, c, length, d_model), has_neighbor

    def forward(self, x: Tensor, channel_mask: Tensor) -> Tuple[Tensor, Tensor]:
        aggregated, has_neighbor = self.aggregate(x, channel_mask)
        output = self.neighbor_projection(aggregated)
        output = self.norm(self.dropout(F.silu(output + self.bias)))
        graph_mask = channel_mask & has_neighbor
        output = output * graph_mask[:, :, None, None].to(dtype=output.dtype)
        return output, graph_mask


class AttentiveStatisticsPooling(nn.Module):
    """Attention-weighted mean and standard deviation over nodes and windows."""

    def __init__(self, d_model: int, attention_hidden: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

    def forward(
        self, x: Tensor, channel_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError("statistics pooling expects [N, C, L, D]")
        n, c, length, d_model = x.shape
        locations = x.reshape(n, c * length, d_model)
        mask = channel_mask[:, :, None].expand(n, c, length).reshape(n, c * length)
        scores = self.score(locations).squeeze(-1)
        attention = _masked_softmax(scores, mask, dim=1)
        mean = torch.sum(attention.unsqueeze(-1) * locations, dim=1)
        variance = torch.sum(
            attention.unsqueeze(-1) * (locations - mean.unsqueeze(1)).square(), dim=1
        )
        std = torch.sqrt(variance.clamp_min(0.0) + self.eps)
        return torch.cat((mean, std), dim=-1), attention, mean, std


class SignalQualityFeatureExtractor(nn.Module):
    """Compute eight scale-aware and scale-free signal-quality descriptors."""

    n_features = 8

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=values.dtype)
        return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def forward(self, x: Tensor, channel_mask: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("quality extractor expects [N, C, T]")
        if channel_mask.shape != x.shape[:2]:
            raise ValueError("quality extractor channel_mask must have shape [N, C]")
        with torch.no_grad():
            work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            signal = x.to(dtype=work_dtype)
            finfo = torch.finfo(work_dtype)

            # Scaling first prevents mean/square reductions from overflowing while
            # retaining log-amplitude as a quality descriptor.
            amplitude_scale = signal.abs().amax(dim=-1, keepdim=True)
            safe_scale = torch.where(
                amplitude_scale > 0.0,
                amplitude_scale,
                torch.ones_like(amplitude_scale),
            )
            scaled = signal / safe_scale
            mean = scaled.mean(dim=-1)
            centered = scaled - mean.unsqueeze(-1)
            rms = torch.sqrt(centered.square().mean(dim=-1).clamp_min(0.0))
            total_rms = torch.sqrt(scaled.square().mean(dim=-1).clamp_min(0.0))
            difference = scaled[..., 1:] - scaled[..., :-1]
            difference_rms = torch.sqrt(
                difference.square().mean(dim=-1).clamp_min(0.0)
            )

            nonzero_rms = rms > 0.0
            log_rms = torch.log(safe_scale.squeeze(-1)) + torch.log(
                rms.clamp_min(finfo.tiny)
            )
            log_rms = torch.where(
                nonzero_rms,
                log_rms,
                torch.full_like(log_rms, math.log(finfo.tiny)),
            )
            roughness = torch.log1p(
                difference_rms / rms.clamp_min(finfo.tiny)
            )
            adaptive_threshold = 0.01 * rms.unsqueeze(-1)
            flat_fraction = (difference.abs() <= adaptive_threshold).float().mean(
                dim=-1
            )
            peak_ratio = torch.log1p(
                centered.abs().amax(dim=-1) / rms.clamp_min(finfo.tiny)
            )
            dc_ratio = mean.abs() / total_rms.clamp_min(finfo.tiny)
            zero_crossing = (
                centered[..., 1:] * centered[..., :-1] < 0
            ).float().mean(dim=-1)

            channel_mean_log_rms = self._masked_mean(log_rms, channel_mask)
            log_rms_variance = self._masked_mean(
                (log_rms - channel_mean_log_rms.unsqueeze(-1)).square(),
                channel_mask,
            )
            masked_log_rms = log_rms.masked_fill(~channel_mask, -torch.inf)
            valid_channel_count = channel_mask.sum(dim=-1).clamp_min(1)
            log_mean_rms = torch.logsumexp(masked_log_rms, dim=-1) - torch.log(
                valid_channel_count.to(dtype=log_rms.dtype)
            )
            low_variance_fraction = self._masked_mean(
                (log_rms < log_mean_rms.unsqueeze(-1) - math.log(20.0)).float(),
                channel_mask,
            )

            features = torch.stack(
                (
                    channel_mean_log_rms,
                    torch.sqrt(log_rms_variance + self.eps),
                    self._masked_mean(roughness, channel_mask),
                    self._masked_mean(flat_fraction, channel_mask),
                    self._masked_mean(peak_ratio, channel_mask),
                    self._masked_mean(dc_ratio, channel_mask),
                    self._masked_mean(zero_crossing, channel_mask),
                    low_variance_fraction,
                ),
                dim=-1,
            )
        return features.float()


class PhysiologicalPredictionHead(nn.Module):
    """Predict band powers, log theta/alpha ratio, and spectral entropy."""

    def __init__(self, segment_dim: int, n_bands: int) -> None:
        super().__init__()
        hidden = max(16, segment_dim // 2)
        self.shared = nn.Sequential(
            nn.LayerNorm(segment_dim),
            nn.Linear(segment_dim, hidden),
            nn.SiLU(),
        )
        self.power = nn.Linear(hidden, n_bands)
        self.log_theta_alpha = nn.Linear(hidden, 1)
        self.entropy = nn.Linear(hidden, 1)

    def forward(self, segment_embedding: Tensor) -> Dict[str, Tensor]:
        hidden = self.shared(segment_embedding)
        relative_power = torch.softmax(self.power(hidden), dim=-1)
        log_ratio = self.log_theta_alpha(hidden)
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
        entropy = torch.sigmoid(self.entropy(hidden))
        return {
            "relative_power": relative_power,
            "log_theta_alpha_ratio": log_ratio,
            "theta_alpha_ratio": ratio,
            "spectral_entropy": entropy,
        }


class SegmentQualityHead(nn.Module):
    """Optional quality prediction for each segment, without batch aggregation."""

    def __init__(
        self,
        segment_dim: int,
        quality_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.quality_norm = nn.LayerNorm(quality_feature_dim)
        self.quality_head = nn.Sequential(
            nn.Linear(segment_dim + quality_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, segment_embedding: Tensor, quality_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if segment_embedding.ndim != 2:
            raise ValueError("segment_embedding must have shape [N, D]")
        if quality_features.ndim != 2:
            raise ValueError("quality_features must have shape [N, Q]")
        if segment_embedding.shape[0] != quality_features.shape[0]:
            raise ValueError("quality features and embeddings must share N")
        normalized_quality = self.quality_norm(quality_features)
        logits = self.quality_head(
            torch.cat((segment_embedding, normalized_quality), dim=-1)
        ).squeeze(-1)
        return logits, torch.sigmoid(logits)


class EEGSegmentClassifier(nn.Module):
    """Complete independent-segment HC/AD classifier.

    The leading dimension is an ordinary batch of shuffled segments.  No
    operation in this class groups samples by subject or mixes different
    segments in the batch.
    """

    class_names = ("HC", "AD")

    def __init__(
        self,
        config: EEGModelConfig,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.config = config
        n_bands = len(config.bands)
        self.filter_bank = FixedBandpassFilterBank(
            config.sampling_rate, config.bands, config.fir_kernel_size
        )
        self.raw_stem = MultiScaleTemporalStem(
            config.d_model,
            config.temporal_kernels,
            config.temporal_stride,
            config.dropout,
        )
        self.band_encoder = BandAttentionEncoder(
            n_bands,
            config.d_model,
            config.temporal_kernels,
            config.temporal_stride,
            config.dropout,
            config.eps,
        )
        self.raw_band_gate = AdaptiveFeatureGate(config.d_model, config.dropout)
        self.channel_embedding = nn.Parameter(
            torch.empty(config.n_channels, config.d_model)
        )
        nn.init.normal_(self.channel_embedding, mean=0.0, std=0.02)
        self.local_temporal = LocalTemporalConv(
            config.d_model,
            config.local_kernel_size,
            config.local_dilation,
            config.dropout,
        )
        self.temporal_ssm = BidirectionalLightSSM(
            config.d_model, config.ssm_state_dim, config.dropout
        )
        self.graph_conv = SparseGraphConv(
            config.n_channels,
            edge_index,
            edge_weight,
            config.d_model,
            config.dropout,
            config.eps,
        )
        self.time_graph_gate = AdaptiveFeatureGate(config.d_model, config.dropout)
        self.segment_pool = AttentiveStatisticsPooling(
            config.d_model, config.attention_hidden, config.eps
        )
        self.segment_projection = nn.Sequential(
            nn.LayerNorm(2 * config.d_model),
            nn.Linear(2 * config.d_model, config.segment_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.segment_classifier = nn.Linear(config.segment_dim, config.num_classes)
        self.physiological_head = PhysiologicalPredictionHead(
            config.segment_dim, n_bands
        )
        self.quality_features = SignalQualityFeatureExtractor(config.eps)
        self.quality_head = SegmentQualityHead(
            config.segment_dim,
            SignalQualityFeatureExtractor.n_features,
            config.quality_hidden,
        )

    def _validate_and_mask_input(
        self,
        eeg: Tensor,
        channel_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if eeg.ndim != 3:
            raise ValueError("eeg must have shape [N, C, T]")
        if not eeg.is_floating_point():
            raise TypeError("eeg must be a floating-point tensor")
        n, c, t = eeg.shape
        if n < 1:
            raise ValueError("eeg must contain at least one segment")
        if eeg.device != self.channel_embedding.device:
            raise ValueError(
                f"eeg is on {eeg.device}, but the model is on "
                f"{self.channel_embedding.device}"
            )
        if c != self.config.n_channels:
            raise ValueError(
                f"model expects {self.config.n_channels} channels, received {c}"
            )
        if t < 2:
            raise ValueError("each segment must contain at least two samples")
        if channel_mask is None:
            channel_mask = torch.ones(n, c, device=eeg.device, dtype=torch.bool)
        else:
            if channel_mask.shape != (n, c):
                raise ValueError("channel_mask must have shape [N, C]")
            channel_mask = channel_mask.to(device=eeg.device, dtype=torch.bool)
        if not channel_mask.any(dim=-1).all():
            raise ValueError("every segment must contain at least one valid channel")

        # Mask before finite checks/encoding so NaN-filled missing channels are safe.
        valid_samples = channel_mask.unsqueeze(-1)
        clean = torch.where(valid_samples, eeg, torch.zeros_like(eeg))
        if not torch.isfinite(clean).all():
            raise ValueError("valid EEG samples must all be finite")
        return clean, channel_mask

    def _normalize(self, eeg: Tensor, channel_mask: Tensor) -> Tensor:
        if not self.config.normalize_per_channel:
            return eeg
        amplitude_scale = eeg.abs().amax(dim=-1, keepdim=True)
        safe_scale = torch.where(
            amplitude_scale > 0.0,
            amplitude_scale,
            torch.ones_like(amplitude_scale),
        )
        scaled = eeg / safe_scale
        mean = scaled.mean(dim=-1, keepdim=True)
        centered = scaled - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.config.eps)
        return normalized * channel_mask.unsqueeze(-1).to(dtype=normalized.dtype)

    def forward(
        self,
        eeg: Tensor,
        channel_mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, Tensor]:
        """Run a segment-level forward pass.

        Args:
            eeg: Flat batch of independent EEG segments ``[N, C, T]``.
            channel_mask: Optional per-segment channel mask ``[N, C]``.
            return_diagnostics: Return large attention/gate tensors when true.
        """

        eeg, channel_mask = self._validate_and_mask_input(eeg, channel_mask)
        n_segments, c, _ = eeg.shape
        quality_features = self.quality_features(eeg, channel_mask)
        normalized_eeg = self._normalize(eeg, channel_mask)
        model_dtype = self.channel_embedding.dtype
        normalized_eeg = normalized_eeg.to(dtype=model_dtype)
        quality_features = quality_features.to(dtype=model_dtype)
        if not torch.isfinite(normalized_eeg).all() or not torch.isfinite(
            quality_features
        ).all():
            raise ValueError(
                "EEG statistics became non-finite after model-dtype conversion"
            )

        fixed_bands = self.filter_bank(normalized_eeg)
        raw_features = self.raw_stem(normalized_eeg)
        band_features, band_attention, local_log_energy = self.band_encoder(fixed_bands)
        fused, raw_band_gate = self.raw_band_gate(raw_features, band_features)

        node_identity = self.channel_embedding[None, :, None, :]
        fused = fused + node_identity
        fused = fused * channel_mask[:, :, None, None].to(dtype=fused.dtype)
        temporal_features = self.local_temporal(fused)
        temporal_features = self.temporal_ssm(temporal_features)
        temporal_features = temporal_features * channel_mask[:, :, None, None].to(
            dtype=temporal_features.dtype
        )

        graph_features, graph_available = self.graph_conv(temporal_features, channel_mask)
        local_features, time_graph_gate = self.time_graph_gate(
            temporal_features, graph_features
        )
        # If masking leaves a node without any neighbor, graph evidence is absent;
        # deterministically fall back to its own temporal representation.
        time_graph_gate = torch.where(
            graph_available[:, :, None, None],
            time_graph_gate,
            torch.ones_like(time_graph_gate),
        )
        local_features = (
            time_graph_gate * temporal_features
            + (1.0 - time_graph_gate) * graph_features
        )
        local_features = local_features * channel_mask[:, :, None, None].to(
            dtype=local_features.dtype
        )

        pooled, local_attention, segment_mean, segment_std = self.segment_pool(
            local_features, channel_mask
        )
        segment_embedding = self.segment_projection(pooled)
        segment_logits = self.segment_classifier(segment_embedding)

        physio = self.physiological_head(segment_embedding)
        learned_quality_logits, learned_quality_scores = self.quality_head(
            segment_embedding, quality_features
        )

        physio_for_loss = torch.cat(
            (
                physio["relative_power"],
                physio["log_theta_alpha_ratio"],
                physio["spectral_entropy"],
            ),
            dim=-1,
        )
        physio_predictions = torch.cat(
            (
                physio["relative_power"],
                physio["theta_alpha_ratio"],
                physio["spectral_entropy"],
            ),
            dim=-1,
        )
        output: Dict[str, Tensor] = {
            "segment_logits": segment_logits,
            "segment_embeddings": segment_embedding,
            "learned_quality_scores": learned_quality_scores,
            "learned_quality_logits": learned_quality_logits,
            "quality_features": quality_features,
            "relative_power_pred": physio["relative_power"],
            "log_theta_alpha_pred": physio["log_theta_alpha_ratio"],
            "theta_alpha_ratio_pred": physio["theta_alpha_ratio"],
            "spectral_entropy_pred": physio["spectral_entropy"],
            "physio_for_loss": physio_for_loss,
            "physio_predictions": physio_predictions,
        }
        if return_diagnostics:
            valid_local = channel_mask[:, :, None, None]
            output.update(
                {
                    "band_attention": band_attention.permute(0, 1, 3, 2)
                    .masked_fill(~valid_local, 0.0),
                    "band_local_log_energy": local_log_energy.permute(0, 1, 3, 2)
                    .masked_fill(~valid_local, 0.0),
                    "raw_band_gate": raw_band_gate.masked_fill(
                        ~valid_local, 0.0
                    ),
                    "time_graph_gate": time_graph_gate.masked_fill(
                        ~valid_local, 0.0
                    ),
                    "local_attention": local_attention.reshape(
                        n_segments, c, local_features.shape[2]
                    ).masked_fill(~channel_mask[:, :, None], 0.0),
                    "segment_local_mean": segment_mean,
                    "segment_local_std": segment_std,
                }
            )
        return output


# Preserve the former import name for callers that construct the model by symbol.
# Its interface and behavior are nevertheless strictly segment-level.
EEGSubjectClassifier = EEGSegmentClassifier


@torch.no_grad()
def compute_neurophysiological_targets(
    eeg: Tensor,
    sampling_rate: float,
    bands: Sequence[Band] = DEFAULT_BANDS,
    channel_mask: Optional[Tensor] = None,
    eps: float = 1.0e-8,
) -> Dict[str, Tensor]:
    """Compute per-segment spectral targets without entering the model graph.

    Relative powers are normalized over the configured bands.  Entropy is
    normalized over all FFT bins from the lowest to highest configured frequency.
    The stable regression target is log(theta/alpha); the ordinary ratio is also
    returned for reporting.
    """

    _validate_bands(bands, sampling_rate)
    if not math.isfinite(eps) or not 0.0 < eps < 0.5:
        raise ValueError("eps must be finite and lie strictly between 0 and 0.5")
    if eeg.ndim != 3 or not eeg.is_floating_point():
        raise ValueError("eeg must be a floating tensor with shape [N, C, T]")
    n, c, t = eeg.shape
    if n < 1 or c < 1:
        raise ValueError("eeg must contain at least one segment and one channel")
    if t < 2:
        raise ValueError("at least two time samples are required")
    if channel_mask is None:
        channel_mask = torch.ones(n, c, device=eeg.device, dtype=torch.bool)
    else:
        if channel_mask.shape != (n, c):
            raise ValueError("channel_mask must have shape [N, C]")
        channel_mask = channel_mask.to(device=eeg.device, dtype=torch.bool)
    if not channel_mask.any(dim=-1).all():
        raise ValueError("every segment must contain at least one valid channel")

    clean = torch.where(channel_mask.unsqueeze(-1), eeg, torch.zeros_like(eeg))
    if not torch.isfinite(clean).all():
        raise ValueError("valid EEG samples must be finite")
    with torch.autocast(device_type=eeg.device.type, enabled=False):
        work_dtype = torch.float64 if clean.dtype == torch.float64 else torch.float32
        signal = clean.to(dtype=work_dtype)

        # Normalize each channel after peak scaling.  This makes all targets
        # invariant to EEG units/amplitude without overflowing on large finite data.
        amplitude_scale = signal.abs().amax(dim=-1, keepdim=True)
        safe_scale = torch.where(
            amplitude_scale > 0.0,
            amplitude_scale,
            torch.ones_like(amplitude_scale),
        )
        scaled = signal / safe_scale
        centered = scaled - scaled.mean(dim=-1, keepdim=True)
        channel_rms = torch.sqrt(centered.square().mean(dim=-1).clamp_min(0.0))
        flat_threshold = 10.0 * torch.finfo(work_dtype).eps
        spectral_channel_mask = channel_mask & (channel_rms > flat_threshold)
        standardized = centered / channel_rms.clamp_min(flat_threshold).unsqueeze(-1)
        standardized = standardized * spectral_channel_mask.unsqueeze(-1).to(
            dtype=standardized.dtype
        )

        # FFT inputs are bounded after standardization, so float32 is both stable
        # and compatible with arbitrary CUDA FFT lengths.
        signal = standardized.float()
        window = torch.hann_window(t, periodic=False, device=eeg.device)
        spectrum = torch.fft.rfft(signal * window, dim=-1)
        psd = spectrum.abs().square()
        frequencies = torch.fft.rfftfreq(t, d=1.0 / sampling_rate).to(eeg.device)

        channel_weights = spectral_channel_mask.float().unsqueeze(-1)
        mean_psd = (psd * channel_weights).sum(dim=1) / channel_weights.sum(
            dim=1
        ).clamp_min(1.0)

        band_powers = []
        for index, (_, low, high) in enumerate(bands):
            if index == len(bands) - 1:
                frequency_mask = (frequencies >= low) & (frequencies <= high)
            else:
                frequency_mask = (frequencies >= low) & (frequencies < high)
            if not frequency_mask.any():
                raise ValueError(
                    f"segment length {t} gives no FFT bin for band {(low, high)}"
                )
            band_powers.append(mean_psd[..., frequency_mask].sum(dim=-1))
        absolute_power = torch.stack(band_powers, dim=-1)
        configured_power = absolute_power.sum(dim=-1)
        total_fft_power = mean_psd.sum(dim=-1)
        physiology_valid_mask = (
            spectral_channel_mask.any(dim=-1)
            & (
                configured_power
                > eps
                * total_fft_power.clamp_min(
                    torch.finfo(total_fft_power.dtype).tiny
                )
            )
        )
        relative_power = absolute_power / configured_power.clamp_min(
            torch.finfo(absolute_power.dtype).tiny
        ).unsqueeze(-1)

        names = [name for name, _, _ in bands]
        if "theta" not in names or "alpha" not in names:
            raise ValueError("theta and alpha bands are required for their ratio")
        theta = absolute_power[..., names.index("theta")]
        alpha = absolute_power[..., names.index("alpha")]
        relative_floor = eps * configured_power
        log_ratio = torch.log(theta + relative_floor) - torch.log(
            alpha + relative_floor
        )
        log_ratio = log_ratio.clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))

        analysis_mask = (frequencies >= bands[0][1]) & (
            frequencies <= bands[-1][2]
        )
        analysis_psd = mean_psd[..., analysis_mask]
        analysis_power = analysis_psd.sum(dim=-1, keepdim=True)
        probability = analysis_psd / analysis_power.clamp_min(
            torch.finfo(analysis_psd.dtype).tiny
        )
        n_bins = analysis_psd.shape[-1]
        if n_bins <= 1:
            raise ValueError("not enough FFT bins to compute spectral entropy")
        entropy = -(
            probability
            * torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny))
        ).sum(dim=-1)
        entropy = entropy / math.log(n_bins)

    relative_power = relative_power.masked_fill(
        ~physiology_valid_mask.unsqueeze(-1), 0.0
    )
    log_ratio = log_ratio.unsqueeze(-1).masked_fill(
        ~physiology_valid_mask.unsqueeze(-1), 0.0
    )
    ratio = ratio.unsqueeze(-1).masked_fill(
        ~physiology_valid_mask.unsqueeze(-1), 0.0
    )
    entropy = entropy.unsqueeze(-1).masked_fill(
        ~physiology_valid_mask.unsqueeze(-1), 0.0
    )
    return {
        "relative_power": relative_power,
        "log_theta_alpha_ratio": log_ratio,
        "theta_alpha_ratio": ratio,
        "spectral_entropy": entropy,
        "physiology_valid_mask": physiology_valid_mask,
        "for_loss": torch.cat((relative_power, log_ratio, entropy), dim=-1),
    }


class EEGMultiTaskLoss(nn.Module):
    """Compose independent segment losses; it performs no optimization."""

    def __init__(
        self,
        segment_weight: float = 1.0,
        physiology_weight: float = 0.20,
        quality_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.segment_weight = segment_weight
        self.physiology_weight = physiology_weight
        self.quality_weight = quality_weight

    def forward(
        self,
        outputs: Dict[str, Tensor],
        segment_labels: Tensor,
        physiology_targets: Optional[Dict[str, Tensor]] = None,
        quality_targets: Optional[Tensor] = None,
        class_weight: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        segment_logits = outputs["segment_logits"]
        if segment_logits.ndim != 2:
            raise ValueError("segment_logits must have shape [N, num_classes]")
        n_segments = segment_logits.shape[0]
        if segment_labels.shape != (n_segments,):
            raise ValueError("segment_labels must have shape [N]")
        segment_labels = segment_labels.to(
            device=segment_logits.device, dtype=torch.long
        )
        if class_weight is not None:
            class_weight = class_weight.to(
                device=segment_logits.device, dtype=segment_logits.dtype
            )

        segment_loss = F.cross_entropy(
            segment_logits, segment_labels, weight=class_weight
        )

        physiology_loss = segment_logits.new_zeros(())
        if physiology_targets is not None:
            physiology_mask = torch.ones(
                n_segments, device=segment_logits.device, dtype=torch.bool
            )
            if "physiology_valid_mask" in physiology_targets:
                target_mask = physiology_targets["physiology_valid_mask"]
                if target_mask.shape != (n_segments,):
                    raise ValueError("physiology_valid_mask must have shape [N]")
                physiology_mask = target_mask.to(
                    device=segment_logits.device, dtype=torch.bool
                )
            if physiology_mask.any():
                physiology_loss = (
                    F.smooth_l1_loss(
                        outputs["relative_power_pred"][physiology_mask],
                        physiology_targets["relative_power"].to(
                            device=segment_logits.device,
                            dtype=outputs["relative_power_pred"].dtype,
                        )[physiology_mask],
                    )
                    + F.smooth_l1_loss(
                        outputs["log_theta_alpha_pred"][physiology_mask],
                        physiology_targets["log_theta_alpha_ratio"].to(
                            device=segment_logits.device,
                            dtype=outputs["log_theta_alpha_pred"].dtype,
                        )[physiology_mask],
                    )
                    + F.smooth_l1_loss(
                        outputs["spectral_entropy_pred"][physiology_mask],
                        physiology_targets["spectral_entropy"].to(
                            device=segment_logits.device,
                            dtype=outputs["spectral_entropy_pred"].dtype,
                        )[physiology_mask],
                    )
                ) / 3.0

        quality_loss = segment_logits.new_zeros(())
        if quality_targets is not None:
            if quality_targets.shape != (n_segments,):
                raise ValueError("quality_targets must have shape [N]")
            target = quality_targets.to(
                device=segment_logits.device, dtype=segment_logits.dtype
            )
            if not torch.isfinite(target).all():
                raise ValueError("quality_targets must be finite")
            if (target < 0).any() or (target > 1).any():
                raise ValueError("quality_targets must lie in [0, 1]")
            quality_loss = F.binary_cross_entropy_with_logits(
                outputs["learned_quality_logits"], target
            )

        total = (
            self.segment_weight * segment_loss
            + self.physiology_weight * physiology_loss
            + self.quality_weight * quality_loss
        )
        return {
            "loss": total,
            "segment_loss": segment_loss,
            "physiology_loss": physiology_loss,
            "quality_loss": quality_loss,
        }


__all__ = [
    "DEFAULT_BANDS",
    "EEGModelConfig",
    "EEGSegmentClassifier",
    "EEGSubjectClassifier",
    "EEGMultiTaskLoss",
    "FixedBandpassFilterBank",
    "SegmentQualityHead",
    "compute_neurophysiological_targets",
    "make_ring_edge_index",
]

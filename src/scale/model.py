from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from x_transformers import Encoder

from scale.lyrics_chorus import ChorusLyricsModel


def lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    return torch.arange(width, device=lengths.device)[None] >= lengths[:, None]


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )
        self.gate = nn.Linear(output_dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class AntiAliasedDownsample(nn.Module):
    def __init__(self, dim: int, stride: int, dropout: float):
        super().__init__()
        kernel = 2 * stride
        # Keep a divisible 25 Hz input at exactly 25/stride Hz.  For the
        # stride-3 (8.333 Hz) SCALE path, padding=1 would silently drop
        # one output frame (e.g. 10500 -> 3499 instead of 3500).
        padding = (stride + 1) // 2
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, stride=stride, padding=padding, groups=dim, bias=False),
            nn.Conv1d(dim, dim, 1, bias=False),
        )
        self.pool = nn.AvgPool1d(stride, stride=stride)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        conv = self.conv(value.transpose(1, 2)).transpose(1, 2)
        residual = self.pool(value.transpose(1, 2)).transpose(1, 2)
        width = min(conv.shape[1], residual.shape[1])
        return self.dropout(F.gelu(self.norm(conv[:, :width] + residual[:, :width])))


class ResidualFunctionDownsample(nn.Module):
    """Residual stride downsampling with a width change."""

    def __init__(self, input_dim: int, output_dim: int, stride: int, dropout: float):
        super().__init__()
        kernel = 2 * stride
        padding = (stride + 1) // 2
        self.depthwise = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=input_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(stride, stride=stride)
        self.residual = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        channels_first = value.transpose(1, 2)
        convolution = self.pointwise(self.depthwise(channels_first))
        residual = self.residual(self.pool(channels_first))
        width = min(convolution.shape[-1], residual.shape[-1])
        value = (convolution[..., :width] + residual[..., :width]).transpose(1, 2)
        return self.dropout(F.gelu(self.norm(value)))


class BiMambaBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int, conv_width: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(d_model=dim, d_state=state_dim, d_conv=conv_width, expand=expand)
        self.backward_mamba = Mamba(d_model=dim, d_state=state_dim, d_conv=conv_width, expand=expand)
        self.mix = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(self, value: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value).masked_fill(padding_mask[..., None], 0.0)
        forward = self.forward_mamba(normalized)
        backward = torch.flip(
            self.backward_mamba(torch.flip(normalized, dims=(1,))), dims=(1,)
        )
        value = value + self.dropout(self.mix(torch.cat([forward, backward], dim=-1)))
        value = value + self.dropout(self.ff(self.ff_norm(value)))
        return value.masked_fill(padding_mask[..., None], 0.0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_mask,
            need_weights=False,
        )
        query = query + self.dropout(attended)
        query = query + self.dropout(self.ff(self.ff_norm(query)))
        return query.masked_fill(query_mask[..., None], 0.0)


class Model(nn.Module):
    """Multi-view fusion -> BiMamba -> global Transformer -> cross-attention heads."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_lyrics = bool(config.get("use_lyrics", True))
        self.feature_names = list(config.feature_specs.keys())
        self.reference_feature = str(config.reference_feature)
        modality_dim = int(config.modality_dim)
        model_dim = int(config.model_dim)
        dropout = float(config.dropout)
        self.modality_dropout = float(config.modality_dropout)
        self.adapters = nn.ModuleDict(
            {
                name: FeatureAdapter(int(spec["dim"]), modality_dim, dropout)
                for name, spec in config.feature_specs.items()
            }
        )
        fusion_dim = modality_dim * (len(self.feature_names) + 1)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.downsample = AntiAliasedDownsample(model_dim, int(config.input_downsample), dropout)
        self.local_blocks = nn.ModuleList(
            BiMambaBlock(
                model_dim,
                int(config.mamba_state_dim),
                int(config.mamba_conv_width),
                int(config.mamba_expand),
                dropout,
            )
            for _ in range(int(config.num_mamba_layers))
        )
        self.global_pool = int(config.global_pool)
        global_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(config.transformer_heads),
            dim_feedforward=int(config.transformer_ff_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            global_layer,
            num_layers=int(config.num_transformer_layers),
            norm=nn.LayerNorm(model_dim),
        )
        self.cross_blocks = nn.ModuleList(
            CrossAttentionBlock(model_dim, int(config.transformer_heads), dropout)
            for _ in range(int(config.num_cross_layers))
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.boundary_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim // 2, 1)
        )
        self.local_boundary_head = nn.Linear(model_dim, 1)
        self.function_feature_names = list(config.function_feature_names)
        function_dropout = float(config.function_dropout)
        function_input_dim = sum(
            int(config.feature_specs[name]["dim"])
            for name in self.function_feature_names
        )
        function_projection_dim = int(config.function_projection_dim)
        function_dim = int(config.function_dim)
        self.function_input_projection = nn.Sequential(
            nn.Linear(function_input_dim, function_projection_dim),
            nn.LayerNorm(function_projection_dim),
            nn.GELU(),
            nn.Dropout(function_dropout),
        )
        self.function_downsample = ResidualFunctionDownsample(
            function_projection_dim,
            function_dim,
            int(config.input_downsample),
            function_dropout,
        )
        self.function_transformer = Encoder(
            dim=function_dim,
            depth=int(config.function_transformer_layers),
            heads=int(config.function_transformer_heads),
            layer_dropout=function_dropout,
            attn_dropout=function_dropout,
            ff_dropout=function_dropout,
            attn_flash=True,
            rotary_pos_emb=True,
        )
        self.function_head = nn.Sequential(
            nn.LayerNorm(function_dim),
            nn.Linear(function_dim, function_dim),
            nn.GELU(),
            nn.Dropout(function_dropout),
            nn.Linear(function_dim, int(config.num_classes)),
        )
        self.lyrics_model = ChorusLyricsModel(config) if self.use_lyrics else None
        chorus_selector = torch.zeros(int(config.num_classes))
        chorus_selector[int(config.lyrics_chorus_label_id)] = 1.0
        self.register_buffer("chorus_selector", chorus_selector, persistent=False)

    def _align_projected(
        self, projected: torch.Tensor, lengths: torch.Tensor, target_lengths: torch.Tensor, target_width: int
    ) -> torch.Tensor:
        output = projected.new_zeros((projected.shape[0], target_width, projected.shape[-1]))
        for index in range(projected.shape[0]):
            source_length = max(1, int(lengths[index].item()))
            target_length = max(1, int(target_lengths[index].item()))
            source = projected[index : index + 1, :source_length].transpose(1, 2)
            aligned = F.interpolate(source, size=target_length, mode="linear", align_corners=False)
            output[index, :target_length] = aligned.transpose(1, 2)[0]
        return output

    def _fuse(self, features: dict, lengths: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_lengths = lengths[self.reference_feature]
        target_width = int(target_lengths.max().item())
        aligned = []
        gate_logits = []
        for name in self.feature_names:
            projected = self.adapters[name](features[name])
            projected = self._align_projected(
                projected, lengths[name], target_lengths, target_width
            )
            aligned.append(projected)
            gate_logits.append(self.adapters[name].gate(projected))
        stacked = torch.stack(aligned, dim=2)
        gates = torch.cat(gate_logits, dim=-1)
        available = torch.ones(
            (stacked.shape[0], len(self.feature_names)), dtype=torch.bool, device=stacked.device
        )
        if self.training and self.modality_dropout > 0:
            available = torch.rand_like(available.float()) >= self.modality_dropout
            empty = ~available.any(dim=1)
            available[empty, self.feature_names.index(self.reference_feature)] = True
        gates = gates.masked_fill(~available[:, None], -torch.inf)
        weights = torch.softmax(gates, dim=-1)
        weighted = (stacked * weights[..., None]).sum(dim=2)
        concatenated = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)
        fused = self.fusion(torch.cat([weighted, concatenated], dim=-1))
        padding_mask = lengths_to_mask(target_lengths, target_width)
        return fused.masked_fill(padding_mask[..., None], 0.0), padding_mask, target_lengths

    def forward(self, batch: dict) -> dict:
        value, input_mask, input_lengths = self._fuse(batch["features"], batch["feature_lengths"])
        value = self.downsample(value)
        output_lengths = torch.div(input_lengths, int(self.config.input_downsample), rounding_mode="floor")
        output_lengths = output_lengths.clamp_min(1).clamp_max(value.shape[1])
        output_mask = lengths_to_mask(output_lengths, value.shape[1])
        value = value.masked_fill(output_mask[..., None], 0.0)
        for block in self.local_blocks:
            value = block(value, output_mask)
        local_value = value
        pooled = F.avg_pool1d(
            value.masked_fill(output_mask[..., None], 0.0).transpose(1, 2),
            kernel_size=self.global_pool,
            stride=self.global_pool,
            ceil_mode=True,
        ).transpose(1, 2)
        global_lengths = torch.div(
            output_lengths + self.global_pool - 1,
            self.global_pool,
            rounding_mode="floor",
        )
        global_mask = lengths_to_mask(global_lengths, pooled.shape[1])
        memory = self.global_transformer(
            pooled, src_key_padding_mask=global_mask
        )
        for block in self.cross_blocks:
            value = block(value, memory, output_mask, global_mask)
        value = self.output_norm(value)

        function_inputs = []
        function_input_width = int(input_lengths.max().item())
        for name in self.function_feature_names:
            function_inputs.append(
                self._align_projected(
                    batch["features"][name],
                    batch["feature_lengths"][name],
                    input_lengths,
                    function_input_width,
                )
            )
        function_value = self.function_input_projection(
            torch.cat(function_inputs, dim=-1)
        )
        function_value = function_value.masked_fill(
            input_mask[..., None], 0.0
        )
        function_value = self.function_downsample(function_value)
        function_value = function_value.masked_fill(
            output_mask[..., None], 0.0
        )
        if function_value.shape[1] != value.shape[1]:
            raise RuntimeError(
                "Independent function and boundary paths produced different widths: "
                f"{function_value.shape[1]} != {value.shape[1]}"
            )
        function_value = self.function_transformer(
            function_value,
            mask=~output_mask,
        )
        function_value = function_value.masked_fill(
            output_mask[..., None], 0.0
        )
        audio_function_logits = self.function_head(function_value)
        result = {
            "boundary_logits": self.boundary_head(value).squeeze(-1),
            "local_boundary_logits": self.local_boundary_head(local_value).squeeze(-1),
            "function_logits": audio_function_logits,
            "audio_function_logits": audio_function_logits,
            "padding_mask": output_mask,
        }
        if self.lyrics_model is not None:
            lyrics = self.lyrics_model(batch, audio_function_logits.shape[1])
            result["function_logits"] = audio_function_logits + (
                lyrics["chorus_delta"][..., None] * self.chorus_selector
            )
            result.update(
                {
                    "lyrics_chorus_logits": lyrics["block_logits"],
                    "lyrics_chorus_frame_logits": lyrics["frame_logits"],
                    "lyrics_frame_mask": lyrics["frame_mask"],
                }
            )
        return result

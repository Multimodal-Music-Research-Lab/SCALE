from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, LongformerModel


def read_soulx_blocks(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    blocks: list[dict] = []
    for segment in payload.get("segments", []):
        candidates = segment.get("lines") or []
        if not candidates and str(segment.get("text") or "").strip():
            candidates = [segment]
        lines = []
        for line in candidates:
            text = str(line.get("text") or "").strip()
            start_ms = line.get("start_ms", segment.get("start_ms"))
            end_ms = line.get("end_ms", segment.get("end_ms"))
            if not text or start_ms is None or end_ms is None:
                continue
            start = float(start_ms) / 1000.0
            end = float(end_ms) / 1000.0
            if np.isfinite(start) and np.isfinite(end) and end > start:
                lines.append({"text": text, "start": start, "end": end})
        if lines:
            blocks.append(
                {
                    "lines": lines,
                    "start": min(line["start"] for line in lines),
                    "end": max(line["end"] for line in lines),
                }
            )
    blocks.sort(key=lambda item: (item["start"], item["end"]))
    return blocks


class ChorusLyricsTokenizer:
    """Tokenize every original SoulX stanza independently."""

    def __init__(
        self,
        model_path: str,
        max_block_tokens: int,
        max_line_tokens: int,
        max_blocks: int,
        max_lines_per_block: int,
        line_token: str = "<LINE>",
        local_files_only: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [line_token]}
        )
        self.line_token_id = int(self.tokenizer.convert_tokens_to_ids(line_token))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.sep_token
        self.max_block_tokens = int(max_block_tokens)
        self.max_line_tokens = int(max_line_tokens)
        self.max_blocks = int(max_blocks)
        self.max_lines_per_block = int(max_lines_per_block)
        self.cls_token_id = int(self.tokenizer.cls_token_id)
        self.sep_token_id = int(self.tokenizer.sep_token_id)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    def empty(self) -> dict:
        return {
            "block_input_ids": [],
            "block_attention_mask": [],
            "line_token_starts": [],
            "line_token_ends": [],
            "block_starts": np.zeros((0,), dtype=np.float32),
            "block_ends": np.zeros((0,), dtype=np.float32),
        }

    def encode(self, path: str | Path, chunk_start_sec: float = 0.0) -> dict:
        encoded = self.empty()
        for block in read_soulx_blocks(path)[: self.max_blocks]:
            ids = [self.cls_token_id]
            line_starts = []
            line_ends = []
            for line in block["lines"][: self.max_lines_per_block]:
                line_ids = self.tokenizer.encode(
                    line["text"], add_special_tokens=False
                )[: self.max_line_tokens]
                if not line_ids or len(ids) + len(line_ids) + 2 > self.max_block_tokens:
                    continue
                ids.append(self.line_token_id)
                line_starts.append(len(ids))
                ids.extend(line_ids)
                line_ends.append(len(ids))
            if not line_starts:
                continue
            ids.append(self.sep_token_id)
            encoded["block_input_ids"].append(np.asarray(ids, dtype=np.int64))
            encoded["block_attention_mask"].append(
                np.ones((len(ids),), dtype=np.int64)
            )
            encoded["line_token_starts"].append(
                np.asarray(line_starts, dtype=np.int64)
            )
            encoded["line_token_ends"].append(
                np.asarray(line_ends, dtype=np.int64)
            )
            encoded["block_starts"] = np.append(
                encoded["block_starts"],
                np.float32(block["start"] - chunk_start_sec),
            )
            encoded["block_ends"] = np.append(
                encoded["block_ends"],
                np.float32(block["end"] - chunk_start_sec),
            )
        return encoded


def pad_lyrics_batch(items: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    batch = len(items)
    block_width = max(1, max(len(item["block_input_ids"]) for item in items))
    token_width = max(
        2,
        max(
            (len(tokens) for item in items for tokens in item["block_input_ids"]),
            default=2,
        ),
    )
    line_width = max(
        1,
        max(
            (len(lines) for item in items for lines in item["line_token_starts"]),
            default=1,
        ),
    )
    input_ids = np.full((batch, block_width, token_width), pad_token_id, dtype=np.int64)
    attention = np.zeros_like(input_ids)
    line_starts = np.zeros((batch, block_width, line_width), dtype=np.int64)
    line_ends = np.zeros_like(line_starts)
    line_mask = np.zeros((batch, block_width, line_width), dtype=bool)
    block_starts = np.zeros((batch, block_width), dtype=np.float32)
    block_ends = np.zeros_like(block_starts)
    block_mask = np.zeros((batch, block_width), dtype=bool)
    for sample, item in enumerate(items):
        for block, tokens in enumerate(item["block_input_ids"]):
            token_count = len(tokens)
            line_count = len(item["line_token_starts"][block])
            input_ids[sample, block, :token_count] = tokens
            attention[sample, block, :token_count] = item["block_attention_mask"][block]
            line_starts[sample, block, :line_count] = item["line_token_starts"][block]
            line_ends[sample, block, :line_count] = item["line_token_ends"][block]
            line_mask[sample, block, :line_count] = True
        blocks = len(item["block_input_ids"])
        if blocks:
            block_starts[sample, :blocks] = item["block_starts"]
            block_ends[sample, :blocks] = item["block_ends"]
            block_mask[sample, :blocks] = True
    return {
        "lyrics_block_input_ids": torch.from_numpy(input_ids),
        "lyrics_block_attention_mask": torch.from_numpy(attention),
        "lyrics_line_token_starts": torch.from_numpy(line_starts),
        "lyrics_line_token_ends": torch.from_numpy(line_ends),
        "lyrics_line_mask": torch.from_numpy(line_mask),
        "lyrics_block_starts": torch.from_numpy(block_starts),
        "lyrics_block_ends": torch.from_numpy(block_ends),
        "lyrics_block_mask": torch.from_numpy(block_mask),
    }


class ChorusLyricsModel(nn.Module):
    """Predict binary chorus evidence from lexical recurrence between stanzas."""

    def __init__(self, config):
        super().__init__()
        encoder_config = AutoConfig.from_pretrained(
            str(config.lyrics_tokenizer_path), local_files_only=True
        )
        encoder_config.attention_window = [
            int(config.lyrics_attention_window)
        ] * encoder_config.num_hidden_layers
        self.encoder = LongformerModel.from_pretrained(
            str(config.lyrics_tokenizer_path),
            config=encoder_config,
            local_files_only=True,
        )
        self.encoder.resize_token_embeddings(int(config.lyrics_vocab_size))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        trainable_layers = int(config.lyrics_longformer_trainable_layers)
        if trainable_layers:
            for layer in self.encoder.encoder.layer[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        hidden = int(config.lyrics_hidden_dim)
        dropout = float(config.lyrics_dropout)
        self.frame_hz = float(config.lyrics_frame_hz)
        self.local_exclusion = int(config.lyrics_recurrence_local_exclusion)
        self.similarity_temperature = float(config.lyrics_similarity_temperature)
        self.block_projection = nn.Sequential(
            nn.LayerNorm(self.encoder.config.hidden_size),
            nn.Linear(self.encoder.config.hidden_size, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.line_projection = nn.Sequential(
            nn.LayerNorm(self.encoder.config.hidden_size),
            nn.Linear(self.encoder.config.hidden_size, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.recurrence_projection = nn.Sequential(
            nn.Linear(2 * hidden + 5, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config.lyrics_transformer_heads),
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.block_transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(config.lyrics_transformer_layers),
            norm=nn.LayerNorm(hidden),
        )
        self.chorus_head = nn.Linear(hidden, 1)
        # Calibrate lyric evidence without conditioning on the dataset source.
        self.lyrics_scale = nn.Parameter(torch.zeros(()))
        self.lyrics_bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _pool_lines(tokens, starts, ends, mask):
        output = tokens.new_zeros((*starts.shape, tokens.shape[-1]))
        for block in range(tokens.shape[0]):
            for line in torch.nonzero(mask[block], as_tuple=False).flatten().tolist():
                start = int(starts[block, line])
                end = int(ends[block, line])
                if end > start:
                    output[block, line] = tokens[block, start:end].mean(dim=0)
        return output

    def _similarity(self, blocks, lines, line_mask, block_mask):
        normalized_blocks = F.normalize(blocks, dim=-1)
        block_similarity = torch.matmul(normalized_blocks, normalized_blocks.transpose(0, 1))
        normalized_lines = F.normalize(lines, dim=-1)
        pairwise = torch.einsum("ild,jmd->ijlm", normalized_lines, normalized_lines)
        pair_valid = line_mask[:, None, :, None] & line_mask[None, :, None, :]
        pairwise = pairwise.masked_fill(~pair_valid, -1e4)
        forward = pairwise.max(dim=-1).values
        backward = pairwise.max(dim=-2).values
        forward = (forward * line_mask[:, None]).sum(dim=-1) / line_mask.sum(dim=-1)[:, None].clamp_min(1)
        backward = (backward * line_mask[None]).sum(dim=-1) / line_mask.sum(dim=-1)[None].clamp_min(1)
        line_similarity = 0.5 * (forward + backward)
        similarity = 0.30 * block_similarity + 0.70 * line_similarity
        index = torch.arange(blocks.shape[0], device=blocks.device)
        separated = (index[:, None] - index[None]).abs() > self.local_exclusion
        valid = block_mask[:, None] & block_mask[None] & separated
        return similarity, valid

    def _recurrence_features(self, blocks, lines, line_mask, block_mask, starts, ends):
        similarity, valid = self._similarity(blocks, lines, line_mask, block_mask)
        masked = similarity.masked_fill(~valid, -1e4)
        weights = torch.softmax(masked / self.similarity_temperature, dim=-1)
        weights = weights.masked_fill(~valid, 0.0)
        context = torch.matmul(weights, blocks)
        maximum = masked.max(dim=-1).values
        maximum = torch.where(valid.any(dim=-1), maximum, torch.zeros_like(maximum))
        soft_count = (torch.sigmoid((similarity - 0.70) / 0.08) * valid).sum(dim=-1)
        best = masked.argmax(dim=-1)
        index = torch.arange(blocks.shape[0], device=blocks.device)
        distance = (best - index).abs() / max(1, blocks.shape[0] - 1)
        song_end = torch.where(block_mask, ends, torch.zeros_like(ends)).max().clamp_min(1.0)
        midpoint = 0.5 * (starts + ends)
        duration = (ends - starts).clamp_min(0.0)
        numeric = torch.stack(
            [maximum, soft_count, distance, midpoint / song_end, duration / song_end],
            dim=-1,
        )
        numeric = numeric.masked_fill(~block_mask[..., None], 0.0)
        return self.recurrence_projection(torch.cat([blocks, context, numeric], dim=-1))

    def _project(self, block_logits, starts, ends, block_mask, width):
        score = block_logits.new_zeros((block_logits.shape[0], width))
        count = block_logits.new_zeros((block_logits.shape[0], width))
        for sample in range(block_logits.shape[0]):
            for block in torch.nonzero(block_mask[sample], as_tuple=False).flatten().tolist():
                left = max(0, min(width, int(torch.floor(starts[sample, block] * self.frame_hz).item())))
                right = max(left + 1, min(width, int(torch.ceil(ends[sample, block] * self.frame_hz).item())))
                if left < width:
                    score[sample, left:right] += block_logits[sample, block]
                    count[sample, left:right] += 1.0
        mask = count > 0
        return score / count.clamp_min(1.0), mask

    def forward(self, batch, width):
        input_ids = batch["lyrics_block_input_ids"]
        attention = batch["lyrics_block_attention_mask"]
        block_mask = batch["lyrics_block_mask"]
        batch_size, blocks, tokens = input_ids.shape
        flat_valid = block_mask.reshape(-1)
        encoded = self.encoder.embeddings.word_embeddings.weight.new_zeros(
            (batch_size * blocks, tokens, self.encoder.config.hidden_size)
        )
        if flat_valid.any():
            flat_ids = input_ids.reshape(batch_size * blocks, tokens)[flat_valid]
            flat_attention = attention.reshape(batch_size * blocks, tokens)[flat_valid]
            global_attention = torch.zeros_like(flat_attention)
            global_attention[:, 0] = 1
            encoded[flat_valid] = self.encoder(
                input_ids=flat_ids,
                attention_mask=flat_attention,
                global_attention_mask=global_attention,
            ).last_hidden_state
        encoded = encoded.reshape(batch_size, blocks, tokens, -1)
        block_states = encoded.new_zeros((batch_size, blocks, self.block_projection[1].out_features))
        line_states = encoded.new_zeros(
            (*batch["lyrics_line_mask"].shape, self.line_projection[1].out_features)
        )
        for sample in range(batch_size):
            valid_blocks = int(block_mask[sample].sum())
            if not valid_blocks:
                continue
            token_mask = attention[sample, :valid_blocks].bool()
            pooled = (encoded[sample, :valid_blocks] * token_mask[..., None]).sum(dim=1)
            pooled = pooled / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
            block_states[sample, :valid_blocks] = self.block_projection(pooled)
            pooled_lines = self._pool_lines(
                encoded[sample, :valid_blocks],
                batch["lyrics_line_token_starts"][sample, :valid_blocks],
                batch["lyrics_line_token_ends"][sample, :valid_blocks],
                batch["lyrics_line_mask"][sample, :valid_blocks],
            )
            line_states[sample, :valid_blocks] = self.line_projection(pooled_lines)
        recurrence = block_states.new_zeros(block_states.shape)
        for sample in range(batch_size):
            recurrence[sample] = self._recurrence_features(
                block_states[sample],
                line_states[sample],
                batch["lyrics_line_mask"][sample],
                block_mask[sample],
                batch["lyrics_block_starts"][sample],
                batch["lyrics_block_ends"][sample],
            )
        safe_block_mask = block_mask.clone()
        safe_block_mask[~safe_block_mask.any(dim=1), 0] = True
        recurrence = self.block_transformer(
            recurrence, src_key_padding_mask=~safe_block_mask
        ).masked_fill(~block_mask[..., None], 0.0)
        block_logits = self.chorus_head(recurrence).squeeze(-1)
        frame_logits, frame_mask = self._project(
            block_logits,
            batch["lyrics_block_starts"],
            batch["lyrics_block_ends"],
            block_mask,
            width,
        )
        delta = self.lyrics_scale * frame_logits + self.lyrics_bias
        delta = delta.masked_fill(~frame_mask, 0.0)
        return {
            "block_logits": block_logits,
            "frame_logits": frame_logits,
            "frame_mask": frame_mask,
            "chorus_delta": delta,
        }

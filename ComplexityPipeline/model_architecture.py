"""Model architecture matching the fine-tuned checkpoint.
This isolates the inference-time network that was used during fine-tuning so
both training and serving can share it without importing the whole fine-tuning
script (which also runs training utilities, logging, etc.).
Only the classes required for inference are included.
"""
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(name: str):
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class EnhancedTemporalEncoder(nn.Module):
    """Bi-directional RNN stack with layer norms and multi-head attention.
    This is a streamlined copy of the encoder used during fine-tuning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnns = nn.ModuleList()
        current_input = input_dim
        for _ in range(num_layers):
            self.rnns.append(
                rnn_cls(
                    current_input,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=0.0,
                )
            )
            current_input = hidden_dim * (2 if bidirectional else 1)

        self.output_dim = current_input
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.output_dim) for _ in range(num_layers)])
        self.attention = nn.MultiheadAttention(self.output_dim, attention_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        for rnn, ln in zip(self.rnns, self.layer_norms):
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_out, _ = rnn(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            else:
                x, _ = rnn(x)
            x = ln(x)

        # Self-attention pooling
        attn_out, _ = self.attention(x, x, x)  # (B, T, D)
        pooled = attn_out.mean(dim=1)  # (B, D)
        return pooled


class StaticFeatureProcessor(nn.Module):
    """Simple MLP for 10-dim static feature vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.3,
        activation: str = "gelu",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        curr = input_dim
        act = get_activation_fn(activation)
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            layers.append(nn.Dropout(dropout_rate))
            curr = h
        layers.append(nn.Linear(curr, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ModalityFusion(nn.Module):
    """Element-wise gated fusion of temporal & static features."""

    def __init__(self, input_dims: List[int]) -> None:
        super().__init__()
        total = sum(input_dims)
        self.gate = nn.Sequential(nn.Linear(total, total), nn.Sigmoid())
        self.total_input_dim = total

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(feats, dim=1)
        return concat * self.gate(concat)


class MultimodalRNN(nn.Module):
    """Final classifier matching the fine-tuned checkpoint."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Extract configuration with exact checkpoint values
        vis_feat_dim = 2048  # From checkpoint: input to temporal_encoder
        static_feat_dim = 10  # From checkpoint analysis
        rnn_hidden_dim = 256  # From checkpoint: LSTM hidden size
        dense_hidden_dim = 128  # From checkpoint analysis
        output_dim = 3  # Easy, Medium, Hard
        
        self.temporal_encoder = EnhancedTemporalEncoder(
            input_dim=vis_feat_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=2,
            rnn_type='lstm',
            bidirectional=True,
            dropout_rate=0.3,
            attention_heads=8
        )
        
        self.static_processor = StaticFeatureProcessor(
            input_dim=static_feat_dim,
            hidden_dims=[dense_hidden_dim],
            output_dim=dense_hidden_dim // 2,
            dropout_rate=0.3,
            batch_norm=True
        )
        
        self.fusion = ModalityFusion([self.temporal_encoder.output_dim, dense_hidden_dim // 2])
        self.prediction_head = StaticFeatureProcessor(
            input_dim=self.fusion.total_input_dim,
            hidden_dims=[dense_hidden_dim],
            output_dim=output_dim,
            dropout_rate=0.3,
            batch_norm=True
        )

    def forward(
        self,
        vis_seq: torch.Tensor,  # (B, T, V)
        static_feats: torch.Tensor,  # (B, 10)
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temp_feat = self.temporal_encoder(vis_seq, lengths)  # (B, D)
        stat_feat = self.static_processor(static_feats)  # (B, d)
        fused = self.fusion([temp_feat, stat_feat])  # (B, total)
        logits = self.prediction_head(fused)  # (B, C)
        return logits

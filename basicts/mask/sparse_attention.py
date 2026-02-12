import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def build_local_causal_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float('-inf'), device=device)
    for i in range(length):
        start = max(0, i - window)
        mask[i, start:i + 1] = 0.0
    return mask


class GraphAnchorSelector(nn.Module):
    def __init__(self, anchor_ratio: float = 0.1, min_anchors: int = 1) -> None:
        super().__init__()
        self.anchor_ratio = anchor_ratio
        self.min_anchors = min_anchors

    def forward(self, patches: torch.Tensor, adp: torch.Tensor) -> torch.Tensor:
        b, n, p, d = patches.shape
        if p == 0:
            return patches.new_zeros(b * n, 0, d)
        anchor_count = max(self.min_anchors, int(math.ceil(p * self.anchor_ratio)))
        anchor_count = min(anchor_count, p)
        importance = adp.mean(dim=0)
        # patches.norm(dim=-1) shape: (b, n, p), importance shape: (n,)
        scores = torch.einsum('bnp,n->bp', patches.norm(dim=-1), importance)
        topk = scores.topk(anchor_count, dim=1).indices
        anchors = []
        for batch_idx in range(b):
            idx = topk[batch_idx]
            anchor_tokens = patches[batch_idx].mean(dim=0)[idx]
            anchor_tokens = anchor_tokens.unsqueeze(0).repeat(n, 1, 1)
            anchors.append(anchor_tokens)
        anchors = torch.cat(anchors, dim=0)
        return anchors


class SparseCausalAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: int, ffn_ratio: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        hidden_dim = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor]) -> torch.Tensor:
        b_n, p, d = x.shape
        device = x.device
        mask = build_local_causal_mask(p, self.window_size, device)
        attn_out, _ = self.local_attn(x, x, x, attn_mask=mask)
        x = x + attn_out
        if anchors is not None and anchors.shape[1] > 0:
            global_out, _ = self.global_attn(x, anchors, anchors)
            x = x + global_out
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class SparseCausalAttentionStack(nn.Module):
    def __init__(self, depth: int, embed_dim: int, num_heads: int, window_size: int, ffn_ratio: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SparseCausalAttentionBlock(embed_dim, num_heads, window_size, ffn_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, anchors: Optional[torch.Tensor]) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out, anchors)
        return out

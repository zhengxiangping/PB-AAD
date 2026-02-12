from typing import List, Tuple

import torch
from torch import nn


class MultiScaleFusion(nn.Module):
    def __init__(self, num_scales: int, embed_dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            for _ in range(num_scales)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, scale_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = 0
        for proj, scale_out in zip(self.projections, scale_outputs):
            pooled = scale_out.mean(dim=2)
            fused = fused + proj(pooled)
        fused = self.output_norm(fused)
        final_scale = scale_outputs[-1][:, :, -1, :]
        return fused, final_scale

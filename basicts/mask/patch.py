import math
import torch
from torch import nn


class TemporalConvPatch(nn.Module):
    """1D temporal convolution to obtain patch tokens."""

    def __init__(self, patch_size: int, in_channel: int, embed_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv1d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.conv(x)
        patches = patches.transpose(1, 2)
        return patches


class MultiScalePatchEmbedding(nn.Module):
    """Hierarchical patch embedding with shared temporal convolutions."""

    def __init__(self, patch_sizes, in_channel: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_sizes = patch_sizes if isinstance(patch_sizes, (list, tuple)) else [patch_sizes]
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleList([
            TemporalConvPatch(patch_size, in_channel, embed_dim)
            for patch_size in self.patch_sizes
        ])

    def forward(self, x: torch.Tensor):
        b, n, c, l = x.shape
        x = x.reshape(b * n, c, l)
        outputs = []
        for embed in self.embeddings:
            patches = embed(x)
            p = patches.shape[1]
            patches = patches.reshape(b, n, p, self.embed_dim)
            outputs.append(patches)
        return outputs

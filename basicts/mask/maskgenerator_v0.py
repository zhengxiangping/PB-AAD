import random
from torch import nn
import torch
class MaskGenerator(nn.Module):
    """Mask generator."""
    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True
    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens
    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens
class AdaptiveMaskGenerator(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
    def forward(self, x):
        B, N, C, L = x.shape
        num_mask = int(L * self.mask_ratio)
        # 计算分数
        score = torch.abs(x)  # [B, N, C, L]
        # topk
        idx = torch.topk(score, num_mask, dim=-1).indices  # [B, N, C, num_mask]
        # 构造mask
        mask = torch.zeros(B, N, C, L, device=x.device)
        # 用scatter_高效赋值
        mask.scatter_(-1, idx, 1)
        return mask
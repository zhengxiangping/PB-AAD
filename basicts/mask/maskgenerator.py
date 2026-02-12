import torch
import torch.nn as nn

class AdaptiveProgressiveMaskGenerator(nn.Module):
    def __init__(self, base_mask_ratio=0.25, final_mask_ratio=0.5, total_epochs=100):
        super().__init__()
        self.base_mask_ratio = base_mask_ratio
        self.final_mask_ratio = final_mask_ratio
        self.total_epochs = total_epochs
        
    def forward(self, x, epoch=0):
        # 随训练进度调整掩码率
        current_ratio = self.base_mask_ratio + (self.final_mask_ratio - self.base_mask_ratio) * \
                        min(1.0, epoch / (self.total_epochs * 0.8))
        
        B, N, C, L = x.shape
        mask = torch.zeros_like(x)
        num_mask = int(L * current_ratio)
        
        # 计算所有元素的重要性分数
        scores = torch.abs(x)  # [B, N, C, L]
        
        # 对每个(b,n,c)找出topk重要位置的索引
        _, indices = torch.topk(scores, k=min(num_mask, L), dim=-1)  # [B, N, C, num_mask]
        
        # 使用scatter创建掩码
        mask.scatter_(-1, indices, 1.0)
        
        return mask
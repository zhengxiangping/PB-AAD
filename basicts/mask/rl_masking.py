import math
from typing import Dict

import torch
from torch import nn


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReinforcementMaskAgent(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_patches: int,
        mask_ratio: float,
        hidden_dim: int = 128,
        state_dim: int = 10,
        baseline_momentum: float = 0.9,
        history_momentum: float = 0.9
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.policy = PolicyNetwork(state_dim, hidden_dim, num_nodes * num_patches)
        self.register_buffer('baseline', torch.tensor(0.0))
        self.baseline_momentum = baseline_momentum
        self.history_momentum = history_momentum
        uniform = torch.full((num_nodes * num_patches,), 1.0 / (num_nodes * num_patches))
        self.register_buffer('mask_history', uniform)
        self.state_dim = state_dim
        self._prior_log = math.log(1.0 / (num_nodes * num_patches))

    def build_state(self, adp: torch.Tensor, prev_adp: torch.Tensor, epoch: int, long_history_data: torch.Tensor) -> torch.Tensor:
        device = adp.device
        centrality = adp.sum(dim=0)
        degree = adp.sum(dim=1)
        centrality_stats = torch.stack([centrality.mean(), centrality.std(), centrality.max()])
        degree_stats = torch.stack([degree.mean(), degree.std(), degree.max()])
        if prev_adp is None:
            variation = torch.tensor(0.0, device=device)
        else:
            diff = adp - prev_adp
            variation = torch.norm(diff, p='fro')
        traffic_energy = long_history_data.abs().mean()
        delta = torch.tensor([
            math.sin(epoch / 10.0),
            math.cos(epoch / 10.0)
        ], device=device)
        state = torch.cat([centrality_stats, degree_stats, variation.view(1), traffic_energy.view(1), delta])
        state = state.unsqueeze(0)
        return state

    def sample_mask(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = state.shape[0]
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        mask_count = max(1, int(self.mask_ratio * probs.shape[-1]))
        indices = torch.multinomial(probs, mask_count, replacement=False)
        selected = probs.gather(1, indices).clamp_min(1e-8)
        log_prob = _safe_log(selected).sum(dim=1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, indices, 1.0)
        token_probs = _safe_normalize(mask)
        history = self.mask_history.unsqueeze(0).expand_as(token_probs)
        history = _safe_normalize(history)
        m = 0.5 * (token_probs + history)
        jsd_part1 = torch.sum(token_probs * (_safe_log(token_probs) - _safe_log(m)), dim=-1)
        jsd_part2 = torch.sum(history * (_safe_log(history) - _safe_log(m)), dim=-1)
        jsd_batch = 0.5 * (jsd_part1 + jsd_part2)
        kl_batch = torch.sum(probs * (_safe_log(probs) - self._prior_log), dim=-1)
        outputs = {
            'mask': mask.view(batch_size, self.num_nodes, self.num_patches),
            'indices': indices,
            'log_prob': log_prob,
            'probs': probs,
            'jsd_batch': jsd_batch,
            'kl_batch': kl_batch,
            'token_probs': token_probs
        }
        self._update_history(token_probs.detach())
        return outputs

    def _update_history(self, token_probs: torch.Tensor) -> None:
        mean_mask = token_probs.mean(dim=0)
        updated = self.history_momentum * self.mask_history + (1 - self.history_momentum) * mean_mask
        self.mask_history = _safe_normalize(updated)

    def update_baseline(self, reward: torch.Tensor) -> None:
        value = reward.mean().detach()
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * value

    def baseline_value(self) -> torch.Tensor:
        return self.baseline.detach()




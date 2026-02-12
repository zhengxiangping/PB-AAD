import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_

from .patch import MultiScalePatchEmbedding
from .sparse_attention import GraphAnchorSelector, SparseCausalAttentionStack
from .multi_scale_fusion import MultiScaleFusion
from .rl_masking import ReinforcementMaskAgent
from ..graphwavenet import GraphWaveNet


class pretrain_model(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        dim: int,
        topK: int,
        adaptive: bool,
        epochs: int,
        patch_size,
        in_channel: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        mask_ratio: float,
        encoder_depth: int,
        decoder_depth: int,
        seq_len: int = 864,
        rl_args: Optional[Dict] = None,
        mode: str = "pre-train"
    ) -> None:
        super().__init__()
        assert topK < num_nodes
        self.num_nodes = num_nodes
        self.dim = dim
        self.topK = topK
        self.adaptive = adaptive
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.patch_sizes = patch_size if isinstance(patch_size, (list, tuple)) else [patch_size]
        self.in_channel = in_channel
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.seq_len = seq_len
        self.mode = mode
        self.rl_args = rl_args or {}
        self.local_window = self.rl_args.get('local_window', 12)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        self.patch_embedding = MultiScalePatchEmbedding(self.patch_sizes, in_channel, embed_dim)
        self.anchor_selector = GraphAnchorSelector(anchor_ratio=self.rl_args.get('anchor_ratio', 0.1))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.encoders = nn.ModuleList([
            SparseCausalAttentionStack(
                encoder_depth,
                embed_dim,
                num_heads,
                max(1, patch // 4),
                mlp_ratio,
                dropout
            )
            for patch in self.patch_sizes
        ])
        self.decoder = SparseCausalAttentionStack(
            max(1, decoder_depth),
            embed_dim,
            num_heads,
            max(1, self.patch_sizes[0] // 4),
            mlp_ratio,
            dropout
        )
        self.fusion = MultiScaleFusion(len(self.patch_sizes), embed_dim)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

        base_patch_count = seq_len // min(self.patch_sizes)
        self.rl_agent = ReinforcementMaskAgent(
            num_nodes=num_nodes,
            num_patches=base_patch_count,
            mask_ratio=mask_ratio,
            hidden_dim=self.rl_args.get('hidden_dim', 128),
            state_dim=self.rl_args.get('state_dim', 10),
            baseline_momentum=self.rl_args.get('baseline_momentum', 0.9),
            history_momentum=self.rl_args.get('history_momentum', 0.9)
        )
        self.rl_coeffs = {
            'lambda_evo': self.rl_args.get('lambda_evo', 1e-3),
            'lambda_div': self.rl_args.get('lambda_div', 1e-3),
            'lambda_rl': self.rl_args.get('lambda_rl', 1e-2),
            'beta_kl': self.rl_args.get('beta_kl', 1e-3)
        }
        self.register_buffer('prev_adp', None, persistent=False)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        trunc_normal_(self.mask_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_adaptive_adj(self) -> torch.Tensor:
        adp = torch.mm(self.nodevec1, self.nodevec2)
        adp = F.relu(adp)
        adp = torch.softmax(adp, dim=1)
        return adp

    def _project_mask(self, base_mask: torch.Tensor, target_tokens: int) -> Optional[torch.Tensor]:
        if base_mask is None:
            return None
        b, n, p_base = base_mask.shape
        if p_base == target_tokens:
            return base_mask
        factor = p_base // target_tokens
        if factor * target_tokens != p_base:
            raise ValueError("Base mask length must be divisible by target tokens.")
        reshaped = base_mask.view(b, n, target_tokens, factor)
        return reshaped.max(dim=-1).values

    def _encode(self, long_history_data: torch.Tensor, epoch: int, adp: torch.Tensor):
        batch_size = long_history_data.shape[0]
        device = long_history_data.device
        long_term = long_history_data.permute(0, 2, 3, 1).contiguous()
        patch_list = self.patch_embedding(long_term)
        labels = patch_list[0]
        mask_outputs = None
        indices = None
        base_mask = None
        if self.mode == "pre-train" and self.training:
            state = self.rl_agent.build_state(adp, self.prev_adp, epoch, long_history_data)
            state = state.expand(batch_size, -1)
            mask_outputs = self.rl_agent.sample_mask(state)
            base_mask = mask_outputs['mask']
            indices = mask_outputs['indices']
        elif self.mode == "pre-train":
            base_mask = torch.zeros(batch_size, self.num_nodes, labels.shape[2], device=device)
        encoded_scales = []
        anchors_cache = []
        for scale_idx, patches in enumerate(patch_list):
            scale_mask = self._project_mask(base_mask, patches.shape[2]) if base_mask is not None else None
            mask_token = self.mask_token.expand(batch_size, self.num_nodes, patches.shape[2], -1)
            if scale_mask is not None:
                patches_masked = patches * (1 - scale_mask.unsqueeze(-1)) + mask_token * scale_mask.unsqueeze(-1)
            else:
                patches_masked = patches
            anchors = self.anchor_selector(patches, adp)
            anchors_cache.append(anchors)
            flat = patches_masked.reshape(batch_size * self.num_nodes, patches.shape[2], self.embed_dim)
            encoded_flat = self.encoders[scale_idx](flat, anchors)
            encoded = encoded_flat.reshape(batch_size, self.num_nodes, patches.shape[2], self.embed_dim)
            encoded_scales.append(encoded)
        fused, final_repr = self.fusion(encoded_scales)
        decoder_input = encoded_scales[0].reshape(batch_size * self.num_nodes, -1, self.embed_dim)
        decoder_out = self.decoder(decoder_input, anchors_cache[0]).reshape_as(encoded_scales[0])
        reconstruction = self.output_layer(decoder_out)
        outputs = {
            'encoded_scales': encoded_scales,
            'fused': fused,
            'final': final_repr,
            'reconstruction': reconstruction,
            'labels': labels,
            'indices': indices,
            'mask_outputs': mask_outputs,
            'base_mask': base_mask
        }
        return outputs

    def forward(self, long_history_data: torch.Tensor, epoch: int):
        adp = self._compute_adaptive_adj()
        prev_adp = self.prev_adp.clone() if isinstance(self.prev_adp, torch.Tensor) else self.prev_adp
        outputs = self._encode(long_history_data, epoch, adp)
        self.prev_adp = adp.detach()
        if self.mode == "pre-train":
            reconstruction = outputs['reconstruction']
            labels = outputs['labels']
            indices = outputs['indices']
            if indices is None:
                raise RuntimeError("Mask indices are required in pre-train mode.")
            batch_size = reconstruction.shape[0]
            flat_recon = reconstruction.reshape(batch_size, -1, self.embed_dim)
            flat_label = labels.reshape(batch_size, -1, self.embed_dim)
            gather_indices = indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            reconstruction_masked = flat_recon.gather(1, gather_indices)
            label_masked = flat_label.gather(1, gather_indices)
            mask_outputs = outputs['mask_outputs']
            sparsity_factor = 0.1 * min(1.0, epoch / (self.epochs * 0.5))
            sparsity_loss = torch.mean(torch.sum(adp, dim=1)) * sparsity_factor
            return {
                'reconstruction': reconstruction_masked,
                'labels': label_masked,
                'sparsity_loss': sparsity_loss,
                'mask_info': mask_outputs,
                'adp': adp,
                'prev_adp': prev_adp
            }
        return outputs['final']


class finetune_model(nn.Module):
    def __init__(self, pre_trained_path: str, mask_args: Dict, backend_args: Dict, backbone_name: str = 'gwnet'):
        super().__init__()
        self.pre_trained_path = pre_trained_path
        self.pretrain_model = pretrain_model(**mask_args)
        self.pretrain_model.mode = "forecasting"
        backend_cls = BACKBONE_FACTORY.get(backbone_name.lower())
        if backend_cls is None:
            raise ValueError(f"Unsupported backbone '{backbone_name}'.")
        self.backend_name = backbone_name.lower()
        self.backend = backend_cls(**backend_args)
        embed_dim = mask_args.get('embed_dim', backend_args.get('fusion_dim', 128))
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.load_pre_trained_model()

    def load_pre_trained_model(self) -> None:
        checkpoint_dict = torch.load(self.pre_trained_path, map_location='cpu')
        try:
            self.pretrain_model.load_state_dict(checkpoint_dict, strict=False)
            print("Loaded pre-trained encoder weights.")
        except RuntimeError as exc:
            print(f"Failed to load pre-trained weights: {exc}")
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            hidden_states = self.pretrain_model(long_history_data, epoch)
        fusion_context = self.fusion_head(hidden_states)
        try:
            y_hat = self.backend(history_data, hidden_states=fusion_context)
        except TypeError as exc:
            raise RuntimeError(f"Backbone '{self.backend_name}' does not accept hidden_states. Please provide an adapter.") from exc
        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        return y_hat






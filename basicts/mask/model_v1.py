import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from .patch import MultiScalePatchEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from ..graphwavenet import GraphWaveNet
from .GIN import GIN_layer
from .maskgenerator import AdaptiveMaskGenerator
class pretrain_model(nn.Module):
    def __init__(self,num_nodes, dim, topK, adaptive, epochs, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout,  mask_ratio, encoder_depth, decoder_depth,  mode="pre-train") -> None:
        super().__init__()
        assert topK < num_nodes
        self.adaptive = adaptive
        self.lamda = 0.8
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.topK = topK
        self.mask_ratio = mask_ratio
        self.selected_feature = 0
        self.mode = mode
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat=None
        self.patch_embedding = MultiScalePatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.mask_generator = AdaptiveMaskGenerator(mask_ratio)
        self.positional_encoding = PositionalEncoding()
        self.GNN_encoder = nn.Sequential(GIN_layer(nn.Linear(1, 32)),
                                        GIN_layer(nn.Linear(32, 1)))
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.GNN_decoder = nn.Sequential(GIN_layer(nn.Linear(embed_dim, 32)),
                                        GIN_layer(nn.Linear(32, embed_dim)))
        self.total_embed_dim = embed_dim * len(patch_size)  # 288
        self.patch_proj = nn.Linear(self.total_embed_dim, self.embed_dim)
        self.output_layer = nn.Linear(embed_dim, embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def encoding(self, long_term_history, epoch, adp, mask=True):
        if mask:
            long_term_history = long_term_history.permute((0, 2, 3, 1))
            patches = self.patch_embedding(long_term_history)
            patches = patches.transpose(-1, -2)
            patches = self.patch_proj(patches)
            batch_size, num_nodes, num_time, num_dim  =  patches.shape
            patches, self.pos_mat = self.positional_encoding(patches)
            Maskg = self.mask_generator(patches)
            mask_patch = Maskg.any(-1)  # [B, N, P]，只要某个特征被mask，该patch就被mask
            masked_token_index = (mask_patch[0, 0] == 1).nonzero(as_tuple=True)[0].tolist()
            unmasked_token_index = (mask_patch[0, 0] == 0).nonzero(as_tuple=True)[0].tolist()
            encoder_input = patches[:, :, unmasked_token_index, :]  # patches shape: [B, N, P, D]
            hidden_states_unmasked = self.encoder(encoder_input)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size,num_nodes, -1, self.embed_dim)
        else:
            batch_size, num_times, num_nodes, _, = long_term_history.shape # B, L, N, C 
            long_term_history = long_term_history.permute((0, 2, 3, 1)) # B, N, C, L 
            patches = self.patch_embedding(long_term_history)     # B, N, d, P
            patches = patches.transpose(-1, -2)         # B, N, P, d
            patches,self.pos_mat = self.positional_encoding(patches) # B, N, P, d
            unmasked_token_index, masked_token_index = None, None
            patches = self.patch_proj(patches)
            encoder_input = patches # B, N, P, d
            hidden_states_unmasked = self.encoder(encoder_input)# B,  P,N, d/# B, N, P, d
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)# B, N, P, d
        return hidden_states_unmasked, unmasked_token_index, masked_token_index
    def decoding(self, hidden_states_unmasked, masked_token_index, adp):
        batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
        unmasked_token_index=[i for i in range(0,len(masked_token_index)+num_time) if i not in masked_token_index ]
        hidden_states_masked = self.pos_mat[:,:,masked_token_index,:]
        hidden_states_masked+=self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
        hidden_states_unmasked+=self.pos_mat[:,:,unmasked_token_index,:]
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full) # B, N, P, d
        hidden_states_full = hidden_states_full.permute((0, 2, 1, 3)) # B, P, N, d
        hidden_states_full, _ = self.GNN_decoder((hidden_states_full, adp))
        hidden_states_full = hidden_states_full.permute((0, 2, 1, 3)) # B, N, P, d
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))
        return reconstruction_full
    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """
        Args:
            reconstruction_full: [B, N, total_P, d]
            real_value_full: [B, N, L, C] (原始输入)
            unmasked_token_index: list
            masked_token_index: list
        Returns:
            reconstruction_masked_tokens: [B, num_masked*d, N]
            label_masked_tokens: [B, num_masked*d, N]
        """
        batch_size, num_nodes, total_P, d = reconstruction_full.shape
        label_full_list = []
        max_patch_num = 0
        max_patch_len = 0
        for p in (self.patch_size if isinstance(self.patch_size, list) else [self.patch_size]):
            label_p = real_value_full.permute(0, 3, 1, 2).unfold(1, p, p)[:, :, :, self.selected_feature, :]  # [B, N, P, L]
            label_p = label_p.permute(0, 2, 1, 3)  # [B, P, N, L]
            label_full_list.append(label_p)
            max_patch_num = max(max_patch_num, label_p.shape[1])
            max_patch_len = max(max_patch_len, label_p.shape[-1])
        label_full_list_aligned = []
        for label_p in label_full_list:
            pad_p = max_patch_num - label_p.shape[1]
            pad_l = max_patch_len - label_p.shape[-1]
            pad = (0, pad_l, 0, 0, 0, pad_p)
            label_p = F.pad(label_p, pad)
            label_p = label_p.permute(0, 2, 1, 3)
            label_full_list_aligned.append(label_p)
        label_full = torch.cat(label_full_list_aligned, dim=1)  # [B, total_P, N, L_max]
        label_full = label_full[..., 0]  # [B, total_P, N]
        label_full = label_full.permute(0, 2, 1)  # [B, N, total_P]
        reconstruction_masked_tokens = reconstruction_full[:, :, masked_token_index, :].contiguous()  # [B, N, num_masked, d]
        label_masked_tokens = label_full[:, :, masked_token_index].contiguous()  # [B, N, num_masked]
        B, N, num_masked, d = reconstruction_masked_tokens.shape
        reconstruction_masked_tokens = reconstruction_masked_tokens.permute(0, 2, 1, 3).reshape(B, num_masked * d, N)
        label_masked_tokens = label_masked_tokens.permute(0, 2, 1).reshape(B, num_masked, N)  # [B, num_masked, N]
        label_masked_tokens = label_masked_tokens.repeat_interleave(d, dim=1)
        return reconstruction_masked_tokens, label_masked_tokens
    def forward(self, long_history_data: torch.Tensor, epoch):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        if self.mode == "pre-train":
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(long_history_data, epoch, adp)
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index, adp)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, long_history_data.permute(0, 2, 3, 1), unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(long_history_data, epoch, adp, mask=False)
            return hidden_states_full

class finetune_model(nn.Module):
    """Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting"""
    def __init__(self, pre_trained_path, mask_args, backend_args):
        super().__init__()
        self.pre_trained_path = pre_trained_path
        self.pretrain_model = pretrain_model(**mask_args)
        self.backend = GraphWaveNet(**backend_args)
        self.load_pre_trained_model()
    def load_pre_trained_model(self):
        checkpoint_dict = torch.load(self.pre_trained_path)
        self.pretrain_model.load_state_dict(checkpoint_dict)
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STDMAE.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """
        short_term_history = history_data     # [B, L, N, 1]
        batch_size, _, num_nodes, _ = history_data.shape
        hidden_states = self.pretrain_model(long_history_data, epoch)
        out_len=1
        hidden_states = hidden_states[:, :, -out_len, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1)
        return y_hat

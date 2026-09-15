import torch
import torch.nn as nn
import torch.nn.functional as F
from Predicept.predictor import GameFormer
from .predictor_modules import *

import torch
import torch.nn as nn
from .predictor_modules import CrossTransformer, FutureEncoder

class ModeSelector(nn.Module):
    def __init__(self, dim=256, num_lat=5, num_lon=12, feature_dim=6):
        super(ModeSelector, self).__init__()
        self.dim = dim
        self.num_lat = num_lat
        self.num_lon = num_lon

        self.lat_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim)
        )

        self.mode_proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU()
        )

        self.neighbor_future_encoder = FutureEncoder()

        self.query_encoder = CrossTransformer(dim=dim)
        self.score_mlp = nn.Sequential(nn.Linear(dim, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, scene_encoding, c_lat, scene_mask=None,
                neighbor_top1_futures=None, neighbor_current_states=None,
                neighbor_valid=None,
                graph_context=None, graph_valid=None, importance=None,
                importance_beta=2.0, hard_topk=None):
        B = scene_encoding.shape[0]
        device = scene_encoding.device

        lat_feat = self.lat_encoder(c_lat)
        lat_feat = torch.max(lat_feat, dim=2)[0]

        j_vals = torch.arange(self.num_lon, dtype=torch.float32, device=device) # [0, 1, ..., 11]
        c_lon_scalar = j_vals / (self.num_lon - 1)

        lon_feat = c_lon_scalar.unsqueeze(1).repeat(1, self.dim)
        lon_feat = lon_feat.unsqueeze(0).expand(B, -1, -1)

        lat_expanded = lat_feat.unsqueeze(2).expand(-1, -1, self.num_lon, -1)
        lon_expanded = lon_feat.unsqueeze(1).expand(-1, self.num_lat, -1, -1)

        combined_modes = torch.cat([lat_expanded, lon_expanded], dim=-1)

        aligned_modes = self.mode_proj(combined_modes)

        mode_queries = aligned_modes.view(B, self.num_lat * self.num_lon, self.dim)

        if graph_context is not None:
            base_context = graph_context                                              # [B, V, D]
            if graph_valid is not None:
                base_mask = ~graph_valid
            elif scene_mask is not None:
                base_mask = scene_mask
            else:
                base_mask = torch.zeros(B, base_context.shape[1], dtype=torch.bool, device=device)

            if importance is not None:
                base_bias = importance_beta * torch.log(importance.clamp_min(1e-6))   # [B, V]
                if (hard_topk is not None) and (not self.training):
                    V = importance.shape[1]
                    k = min(hard_topk, V)
                    keep_idx = importance.topk(k, dim=1).indices                      # [B, k]
                    keep = torch.zeros(B, V, dtype=torch.bool, device=device)
                    keep.scatter_(1, keep_idx, True)
                    base_mask = base_mask | (~keep)
            else:
                base_bias = torch.zeros(B, base_context.shape[1], device=device)
        else:
            base_context = scene_encoding
            base_mask = scene_mask if scene_mask is not None else \
                torch.zeros(B, scene_encoding.shape[1], dtype=torch.bool, device=device)
            base_bias = torch.zeros(B, base_context.shape[1], device=device)

        context = base_context
        full_mask = base_mask
        full_bias = base_bias

        attn_bias = None
        key_padding = full_mask
        if (importance is not None) and (graph_context is not None):
            neg_inf = torch.finfo(full_bias.dtype).min
            full_bias = full_bias.masked_fill(full_mask, neg_inf)
            L = mode_queries.shape[1]
            S_tot = context.shape[1]
            n_heads = self.query_encoder.cross_attention.num_heads
            attn_bias = full_bias[:, None, :].expand(B, L, S_tot)
            attn_bias = attn_bias.unsqueeze(1).expand(B, n_heads, L, S_tot).reshape(B * n_heads, L, S_tot).contiguous()
            key_padding = None

        mode_features = self.query_encoder(mode_queries, context, context, mask=key_padding, attn_bias=attn_bias)  # [B, 60, D]
        
        mode_scores = self.score_mlp(mode_features).squeeze(-1) # [B, 60]

        if not self.training:
            lat_valid_mask = (torch.abs(c_lat).sum(dim=(2, 3)) > 1e-4)              # [B, 5]
            mode_valid_mask = lat_valid_mask.unsqueeze(2).expand(-1, -1, self.num_lon)  # [B, 5, 12]
            mode_valid_mask = mode_valid_mask.reshape(B, self.num_lat * self.num_lon)   # [B, 60]
            mode_scores = mode_scores.masked_fill(~mode_valid_mask, -1e9)

        return mode_scores, mode_features

    


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_modules import FutureEncoder, GMMPredictor, CrossTransformer
from .relevance_graph import (
    PolylineEncoder, build_edge_features, _polyline_pose_and_valid, EDGE_FEATURE_DIM,
    NODE_TYPE_EGO, NODE_TYPE_VEHICLE, NODE_TYPE_PEDESTRIAN, NODE_TYPE_BICYCLE,
)
from .channels import (compute_channels, compute_map_channels, select_ego_corridor,
                       NUM_CHANNELS, NUM_EVIDENCE, NUM_MAP_CHANNELS, NUM_MAP_EVIDENCE,
                       CH_COLLISION_COURSE, CH_SHARES_INTERSECTION, CH_MERGES,
                       CH_FOLLOWS, CH_OVERTAKES)
from .channels_v3 import (compute_agent_channels as compute_agent_channels_v3, compute_map_channels_v3,
                          NUM_A as NUM_CHANNELS_V3, NUM_M as NUM_MAP_CHANNELS_V3, _point_in_polys)
from .decision_labels import (LON5_MAP, LAT5_MAP, NUM_LON5, NUM_LAT5,
                              LON5_FAMILY, LAT5_FAMILY, NUM_FAMILIES,
                              LON4_MAP, LAT5V_MAP, NUM_LON4, NUM_LAT5V, LAT5L_MAP,
                              LON_CE_WEIGHT, LAT_CE_WEIGHT, LON_MERGED_CE_WEIGHT,
                              LON5_CE_WEIGHT, LAT5_CE_WEIGHT, LON4_CE_WEIGHT,
                              LAT5V_CE_WEIGHT, LAT5L_CE_WEIGHT, NUM_LON_MERGED)

NUM_AGENT_TYPES = 4
CHANNEL_SETS = {'v2': (NUM_CHANNELS, NUM_MAP_CHANNELS), 'v3': (NUM_CHANNELS_V3, NUM_MAP_CHANNELS_V3)}
EGO_CONCEPT_NAMES = ['speed', 'isStopped', 'isDecelerating', 'isAccelerating', 'entersJunction']
NUM_EGO_CONCEPTS = len(EGO_CONCEPT_NAMES)
EGO_STOPPED_MPS, EGO_ACC_MPS2 = 0.30, 0.30
EGO_PRED_NAMES = ['isStationary', 'isBraking', 'isSteady', 'isAccelerating', 'routeTurnsLeft', 'routeStraight', 'routeTurnsRight']
NUM_EGO_PREDS = len(EGO_PRED_NAMES)
EGO_STAT_MPS, EGO_ACC_PRED_MPS2 = 0.5, 0.5
ROUTE_LOOK_M, ROUTE_TURN_RAD = 40.0, math.radians(30.0)


def route_turn(ref_path):
    B = ref_path.shape[0]
    if ref_path.shape[1] > 1:
        sel = select_ego_corridor(ref_path); xy = ref_path[torch.arange(B, device=ref_path.device), sel, :, :2]
    else:
        xy = ref_path[:, 0, :, :2]
    xy = xy.float(); pv = xy.abs().sum(-1) > 1e-6
    seg = (xy[:, 1:] - xy[:, :-1]); segv = pv[:, 1:] & pv[:, :-1]
    s = torch.cat([torch.zeros_like(seg[:, :1, 0]), (seg.norm(dim=-1) * segv.float()).cumsum(-1)], dim=1)
    h = torch.atan2(seg[..., 1], seg[..., 0])                                                                # [B,P-1]
    n_seg = segv.float().sum(-1).long().clamp(min=1)
    i0 = torch.zeros(B, dtype=torch.long, device=xy.device)
    i1 = ((s[:, 1:] <= ROUTE_LOOK_M) & segv).float().sum(-1).long().clamp(min=1) - 1
    i1 = torch.minimum(i1, n_seg - 1)
    d = h.gather(1, i1[:, None])[:, 0] - h.gather(1, i0[:, None])[:, 0]
    return torch.atan2(torch.sin(d), torch.cos(d))


def ego_predicates(ego_past, ref_path=None):
    v = ego_past[..., 3:5].norm(dim=-1)
    v_now = v[:, -1]
    a = (v[:, -1] - v[:, -6]) / 0.5 if v.shape[1] >= 6 else torch.zeros_like(v_now)
    brk = a <= -EGO_ACC_PRED_MPS2
    acc = a >= EGO_ACC_PRED_MPS2
    stat = ~brk & ~acc & (v_now < EGO_STAT_MPS)
    steady = ~brk & ~acc & ~stat
    if ref_path is None:
        d = torch.zeros_like(v_now)
    else:
        d = route_turn(ref_path)
    left = d > ROUTE_TURN_RAD; right = d < -ROUTE_TURN_RAD; straight = ~left & ~right
    return torch.stack([stat, brk, steady, acc, left, straight, right], dim=-1)[:, None].detach()


def ego_concept_features(ego_past, ref_path=None, intersections=None):
    v = ego_past[..., 3:5].norm(dim=-1)                                   # [B,T]
    v_now = v[:, -1]
    a = (v[:, -1] - v[:, -6]) / 0.5 if v.shape[1] >= 6 else torch.zeros_like(v_now)
    enters = torch.zeros_like(v_now)
    if ref_path is not None and intersections is not None:
        B = ref_path.shape[0]
        if ref_path.shape[1] > 1:
            sel = select_ego_corridor(ref_path)
            xy = ref_path[torch.arange(B, device=ref_path.device), sel, :, :2]
        else:
            xy = ref_path[:, 0, :, :2]
        xy = xy.float()
        pv = xy.abs().sum(-1) > 1e-6                                       # [B,P]
        seg = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) * (pv[:, 1:] & pv[:, :-1]).float()
        s = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], dim=1)
        lon_ahead = (v_now * 5.0).clamp(min=25.0, max=80.0)[:, None]
        ahead = pv & (s > 0) & (s <= lon_ahead)
        poly = intersections.float()
        ppv = poly[..., :2].abs().sum(-1) > 1e-6                            # [B,K,P]
        polv = ppv.sum(-1) >= 3                                             # [B,K]
        if bool(polv.any()):
            pts = torch.cat([torch.zeros(B, 1, 2, device=xy.device, dtype=xy.dtype), xy], dim=1)
            inside = _point_in_polys(pts, poly, ppv, polv).any(-1)          # [B,1+P]
            now_in = inside[:, 0]
            enters = ((inside[:, 1:] & ahead).any(-1) & ~now_in).float()
    return torch.stack([v_now / 10.0, (v_now <= EGO_STOPPED_MPS).float(),
                        (a <= -EGO_ACC_MPS2).float(), (a >= EGO_ACC_MPS2).float(), enters], dim=-1).detach()
CONFLICT_FEATURE_DIM = 4
UNRELIABLE_CHANNELS = (CH_COLLISION_COURSE, CH_SHARES_INTERSECTION, CH_MERGES)
L1_PROMOTED_CHANNELS = (CH_FOLLOWS, CH_MERGES, CH_OVERTAKES)


def _conflict_features(neighbor_futures, neighbor_states, route_xy, route_valid, ego_speed,
                       ref_xy=None, ref_valid=None, dt=0.1, aligned_mode='straight'):
    B, N, T, _ = neighbor_futures.shape
    dev = neighbor_futures.device
    BIG, CAP = 1e6, 100.0
    fut_valid = neighbor_futures.abs().sum(-1) > 1e-6

    if route_xy.shape[1] > 0:
        d = torch.cdist(neighbor_futures.reshape(B, N * T, 2), route_xy)    # [B,N*T,P]
        d = d.masked_fill(~route_valid[:, None, :], BIG)
        d_route = d.min(-1).values.reshape(B, N, T)
    else:
        d_route = torch.full((B, N, T), BIG, device=dev)
    d_route = d_route.masked_fill(~fut_valid, BIG).min(-1).values           # [B,N]

    t_idx = torch.arange(1, T + 1, device=dev, dtype=neighbor_futures.dtype)
    s_t = ego_speed[:, None] * dt * t_idx[None, :]
    if aligned_mode == 'arc' and ref_xy is not None and ref_xy.shape[1] > 1:
        seg = torch.norm(ref_xy[:, 1:] - ref_xy[:, :-1], dim=-1)            # [B,R-1]
        seg = seg * (ref_valid[:, 1:] & ref_valid[:, :-1]).to(seg.dtype)
        cum = torch.cat([torch.zeros(B, 1, device=dev, dtype=seg.dtype), seg.cumsum(-1)], dim=1)
        pick = (cum[:, None, :] - s_t[:, :, None]).abs()                    # [B,T,R]
        pick = pick.masked_fill(~ref_valid[:, None, :], BIG)
        idx = pick.argmin(-1)                                               # [B,T]
        ego_pos = torch.gather(ref_xy, 1, idx[..., None].expand(-1, -1, 2))  # [B,T,2]
    else:
        ego_pos = torch.stack([s_t, torch.zeros_like(s_t)], dim=-1)
    d_align = torch.norm(neighbor_futures - ego_pos[:, None], dim=-1)       # [B,N,T]
    d_ego_aligned = d_align.masked_fill(~fut_valid, BIG).min(-1).values     # [B,N]

    corr_xy = ref_xy if ref_xy is not None else route_xy
    corr_v = ref_valid if ref_valid is not None else route_valid
    if corr_xy is not None and corr_xy.shape[1] > 0:
        reach = (ego_speed * dt * T).clamp(min=10.0)                        # [B]
        reach_v = corr_v & (torch.norm(corr_xy, dim=-1) <= reach[:, None])
        d_r = torch.cdist(neighbor_futures.reshape(B, N * T, 2), corr_xy)   # [B,N*T,R]
        d_r = d_r.masked_fill(~reach_v[:, None, :], BIG)
        d_ego_spatial = d_r.min(-1).values.reshape(B, N, T)
        d_ego_spatial = d_ego_spatial.masked_fill(~fut_valid, BIG).min(-1).values   # [B,N]
    else:
        d_ego_spatial = torch.full((B, N), BIG, device=dev)

    d0 = torch.norm(neighbor_states[..., :2], dim=-1)                       # [B,N]
    approaching = (d0 - d_ego_aligned).clamp(-CAP, CAP)                     # [B,N]

    feats = torch.stack([
        torch.log1p(d_route.clamp(max=CAP)),
        torch.log1p(d_ego_aligned.clamp(max=CAP)),
        torch.log1p(d_ego_spatial.clamp(max=CAP)),
        approaching / 10.0,
    ], dim=-1)                                                             # [B,N,4]
    return feats.detach()


def _agent_types(neighbor_agents_past, num_neighbors):
    B = neighbor_agents_past.shape[0]
    device = neighbor_agents_past.device
    onehot = neighbor_agents_past[:, :num_neighbors, -1, 8:11]        # [B, N, 3]
    nbr_type = onehot.argmax(-1) + NODE_TYPE_VEHICLE                  # 0->1,1->2,2->3
    ego_type = torch.full((B, 1), NODE_TYPE_EGO, dtype=torch.long, device=device)
    return torch.cat([ego_type, nbr_type.long()], dim=1)             # [B, 1+N]


class _EdgeMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden = out_dim * hidden_mult
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return self.dropout(h)


class _FFN(nn.Module):

    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.w3 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.w2(self.act(self.w1(h)) * self.w3(h))
        return self.dropout(h) + residual


class _NbrMapEnrichLayer(nn.Module):

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.We_k = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.We_v = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.attn = nn.Parameter(torch.empty(heads, self.dh)); nn.init.xavier_uniform_(self.attn)
        self.norm = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, dropout=dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h_nbr, h_map, edge_nbr_map, map_valid):
        B, N = h_nbr.shape[0], h_nbr.shape[1]
        S = h_map.shape[1]
        H, dh = self.heads, self.dh
        q = self.Wq(h_nbr).view(B, N, 1, H, dh)
        k = self.Wk(h_map).view(B, 1, S, H, dh)
        ek = self.We_k(edge_nbr_map).view(B, N, S, H, dh)
        v = self.Wv(h_map).view(B, 1, S, H, dh) + self.We_v(edge_nbr_map).view(B, N, S, H, dh)
        s = self.leaky(q + k + ek)                                      # [B,N,S,H,dh]
        a = (s * self.attn).sum(-1)                                     # [B,N,S,H]
        invalid = ~map_valid[:, None, :, None]                          # [B,1,S,1]
        M = torch.softmax(a.masked_fill(invalid, torch.finfo(a.dtype).min), dim=2).masked_fill(invalid, 0.0)
        ctx = (M.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, H * dh)
        return self.ffn(self.norm(h_nbr + ctx))


class EgoCausalLayer(nn.Module):

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1, gate='softmax',
                 ag_edge_dim=None, conflict_bias=False, typed_kv=False, map_edge_dim=None,
                 n_ch=NUM_CHANNELS, n_mch=NUM_MAP_CHANNELS, ego_family=False, n_ego=NUM_EGO_PREDS):
        super().__init__()
        assert dim % heads == 0
        self.n_ch, self.n_mch = int(n_ch), int(n_mch)
        self.ego_family = bool(ego_family)
        self.n_ego = int(n_ego)
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.gate = gate
        self.typed_kv = bool(typed_kv)
        if self.typed_kv:
            assert gate == 'softmax', "typed_kv only supports the softmax gate"
            self.Wk_ch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_ch)])
            self.Wv_ch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_ch)])
            self.attn_cas_ch = nn.Parameter(torch.empty(self.n_ch, heads, self.dh))
            self.attn_cfd_ch = nn.Parameter(torch.empty(self.n_ch, heads, self.dh))
            self.Wk_mch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_mch)])
            self.Wv_mch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_mch)])
            self.attn_cas_mch = nn.Parameter(torch.empty(self.n_mch, heads, self.dh))
            self.attn_cfd_mch = nn.Parameter(torch.empty(self.n_mch, heads, self.dh))
            for p in (self.attn_cas_ch, self.attn_cfd_ch, self.attn_cas_mch, self.attn_cfd_mch):
                nn.init.xavier_uniform_(p)
        if self.ego_family:
            assert self.typed_kv, 'ego_family requires typed_kv'
            self.Wq_ego = nn.Linear(dim, dim)
            self.Wk_ech = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_ego)])
            self.Wv_ech = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_ego)])
            self.attn_cas_ech = nn.Parameter(torch.empty(self.n_ego, heads, self.dh))
            self.attn_cfd_ech = nn.Parameter(torch.empty(self.n_ego, heads, self.dh))
            nn.init.xavier_uniform_(self.attn_cas_ech); nn.init.xavier_uniform_(self.attn_cfd_ech)
            self.ego_edge_k = nn.Parameter(torch.zeros(dim))
            self.ego_edge_v = nn.Parameter(torch.zeros(dim))
            self.ego_family_bias = nn.Parameter(torch.zeros(1))
        self.conflict_bias = conflict_bias
        if conflict_bias:
            self.conflict_w = nn.Parameter(torch.full((3,), -1.0))
        ag_edge_dim = edge_dim if ag_edge_dim is None else ag_edge_dim
        self.Wq_ag = nn.Linear(dim, dim)
        self.Wk_ag = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])
        self.Wv_ag = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])
        self.We_k_ag = _EdgeMLP(ag_edge_dim, dim, dropout=dropout); self.We_v_ag = _EdgeMLP(ag_edge_dim, dim, dropout=dropout)
        self.attn_cas = nn.Parameter(torch.empty(heads, self.dh))
        self.attn_cfd = nn.Parameter(torch.empty(heads, self.dh))
        self.Wq_mp = nn.Linear(dim, dim); self.Wk_mp = nn.Linear(dim, dim); self.Wv_mp = nn.Linear(dim, dim)
        map_edge_dim = edge_dim if map_edge_dim is None else map_edge_dim
        self.We_k_mp = _EdgeMLP(map_edge_dim, dim, dropout=dropout); self.We_v_mp = _EdgeMLP(map_edge_dim, dim, dropout=dropout)
        self.attn_cas_mp = nn.Parameter(torch.empty(heads, self.dh))
        self.attn_cfd_mp = nn.Parameter(torch.empty(heads, self.dh))
        for p in (self.attn_cas, self.attn_cfd, self.attn_cas_mp, self.attn_cfd_mp):
            nn.init.xavier_uniform_(p)
        self.gate_bias = nn.Parameter(torch.full((4,), -2.0))
        self.joint_softmax = False
        self.self_fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.out_fc_cas = nn.Linear((4 if self.ego_family else 3) * dim, dim)
        self.out_fc_cfd = nn.Linear((4 if self.ego_family else 3) * dim, dim)
        self.norm_cas = nn.LayerNorm(dim)
        self.norm_cfd = nn.LayerNorm(dim)
        self.ego_residual = True
        self.ffn_cas = _FFN(dim, dropout=dropout)
        self.ffn_cfd = _FFN(dim, dropout=dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, q1, k, ek, msg, valid, attn_cas, attn_cfd, bias, conflict_bias=None):
        B = k.shape[0]
        s = self.leaky(q1 + k + ek)                          # [B,Nk,H,dh]
        a_cas = (s * attn_cas).sum(-1)                        # [B,Nk,H]
        a_cfd = (s * attn_cfd).sum(-1)
        if conflict_bias is not None:
            a_cas = a_cas - conflict_bias[:, :, None]
        invalid = ~valid[:, :, None]                          # [B,Nk,1]
        neg_inf = torch.finfo(a_cas.dtype).min
        if self.gate == 'sigmoid':
            M_cas_h = torch.sigmoid(a_cas + bias[0]).masked_fill(invalid, 0.0)
            M_cfd_h = torch.sigmoid(a_cfd + bias[1]).masked_fill(invalid, 0.0)
            den_cas = M_cas_h.sum(dim=1).clamp(min=1.0)[:, None, :, None]      # [B,1,H,1]
            den_cfd = M_cfd_h.sum(dim=1).clamp(min=1.0)[:, None, :, None]
            cas = ((M_cas_h.unsqueeze(-1) * msg) / den_cas).sum(dim=1).reshape(B, self.dim)
            cfd = ((M_cfd_h.unsqueeze(-1) * msg) / den_cfd).sum(dim=1).reshape(B, self.dim)
        else:
            M_cas_h = torch.softmax(a_cas.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
            M_cfd_h = torch.softmax(a_cfd.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
            has_any = valid.any(dim=1)                                            # [B]
            if not bool(has_any.all()):
                z = has_any[:, None, None].float()
                M_cas_h = torch.nan_to_num(M_cas_h) * z
                M_cfd_h = torch.nan_to_num(M_cfd_h) * z
            if getattr(self, 'uniform_mask', False):
                u = valid.float()[:, :, None].expand_as(M_cas_h)
                M_cas_h = u / u.sum(dim=1, keepdim=True).clamp(min=1.0)
            cas = (M_cas_h.unsqueeze(-1) * msg).sum(dim=1).reshape(B, self.dim)   # [B,D]
            cfd = (M_cfd_h.unsqueeze(-1) * msg).sum(dim=1).reshape(B, self.dim)

        eps = 1e-12
        M_cas_mean = M_cas_h.mean(-1)                                              # [B,Nk]
        M_cfd_mean = M_cfd_h.mean(-1)
        if self.gate == 'sigmoid':
            pc = M_cas_h / M_cas_h.sum(dim=1, keepdim=True).clamp(min=eps)
            pf = M_cfd_h / M_cfd_h.sum(dim=1, keepdim=True).clamp(min=eps)
        else:
            pc, pf = M_cas_h, M_cfd_h
        ent_cas_mean = -(M_cas_mean.clamp(min=eps).log() * M_cas_mean).sum(-1)
        ent_cas_headmean = (-(pc.clamp(min=eps).log() * pc).sum(1)).mean(-1)   # [B]
        ent_cfd_mean = -(M_cfd_mean.clamp(min=eps).log() * M_cfd_mean).sum(-1)
        ent_cfd_headmean = (-(pf.clamp(min=eps).log() * pf).sum(1)).mean(-1)

        n_valid = valid.sum(-1).clamp(min=1).float()                                # [B]
        log_n = n_valid.clamp(min=2).log()
        ent_cas_mean = ent_cas_mean / log_n
        ent_cas_headmean = ent_cas_headmean / log_n
        ent_cfd_mean = ent_cfd_mean / log_n
        ent_cfd_headmean = ent_cfd_headmean / log_n

        return (cas, cfd, M_cas_mean, M_cfd_mean,
                ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean)

    def _typed_logits(self, q1, h_src, ek, edge_v, entry_valid, Wk_list, Wv_list,
                      attn_cas_ch, attn_cfd_ch, conflict_bias=None):
        B, S, D = h_src.shape
        H, dh = self.heads, self.dh
        R = entry_valid.shape[-1]
        k_ch = torch.stack([m(h_src) for m in Wk_list], dim=2).view(B, S, R, H, dh)
        v_ch = torch.stack([m(h_src) for m in Wv_list], dim=2).view(B, S, R, H, dh)
        msg = v_ch + edge_v[:, :, None]
        s_ = self.leaky(q1[:, :, None] + k_ch + ek[:, :, None])           # [B,S,R,H,dh]
        a_cas = (s_ * attn_cas_ch[None, None]).sum(-1)                    # [B,S,R,H]
        a_cfd = (s_ * attn_cfd_ch[None, None]).sum(-1)
        if conflict_bias is not None:
            a_cas = a_cas - conflict_bias[:, :, None, None]
        return a_cas, a_cfd, msg

    def _attend_typed_joint(self, *fams):
        outs, flat_cas, flat_cfd, shapes = [], [], [], []
        for i, (a_cas, a_cfd, msg, ev) in enumerate(fams):
            B, S, R, H = a_cas.shape
            inv = ~ev[..., None]
            neg_inf = torch.finfo(a_cas.dtype).min
            fb = self.family_bias[i] if i < 2 else self.ego_family_bias[0]
            flat_cas.append((a_cas + fb).masked_fill(inv, neg_inf).view(B, S * R, H))
            flat_cfd.append((a_cfd + fb).masked_fill(inv, neg_inf).view(B, S * R, H))
            shapes.append((B, S, R, H, msg, ev))
        cat_cas = torch.cat(flat_cas, dim=1)                              # [B, (S*R)_ag + (S*R)_mp, H]
        cat_cfd = torch.cat(flat_cfd, dim=1)
        M_cas = torch.softmax(cat_cas, dim=1)
        M_cfd = torch.softmax(cat_cfd, dim=1)
        has_any = torch.cat([f[3].view(f[3].shape[0], -1) for f in fams], dim=1).any(dim=1)
        z = has_any[:, None, None].float()
        M_cas = torch.nan_to_num(M_cas) * z
        M_cfd = torch.nan_to_num(M_cfd) * z
        if getattr(self, 'uniform_mask', False):
            valid_all = torch.cat([f[3].reshape(f[3].shape[0], -1) for f in fams], dim=1)   # [B, n]
            u = valid_all.float()[:, :, None].expand_as(M_cas)
            M_cas = u / u.sum(dim=1, keepdim=True).clamp(min=1.0)
        off = 0
        for (B, S, R, H, msg, ev) in shapes:
            n = S * R
            mc = M_cas[:, off:off + n].view(B, S, R, H).masked_fill(~ev[..., None], 0.0)
            mf = M_cfd[:, off:off + n].view(B, S, R, H).masked_fill(~ev[..., None], 0.0)
            off += n
            cas = (mc[..., None] * msg).sum(dim=(1, 2)).reshape(B, self.dim)
            cfd = (mf[..., None] * msg).sum(dim=(1, 2)).reshape(B, self.dim)
            w_src = mc.sum(dim=2, keepdim=True).clamp(min=1e-6)                # [B,S,H]
            wf_src = mf.sum(dim=2, keepdim=True).clamp(min=1e-6)
            src_cas = ((mc / w_src)[..., None] * msg).sum(dim=2).reshape(B, S, self.dim)
            src_cfd = ((mf / wf_src)[..., None] * msg).sum(dim=2).reshape(B, S, self.dim)
            outs.append((cas, cfd, mc.mean(-1), mf.mean(-1), src_cas, src_cfd))
        return outs

    def _attend_typed(self, q1, h_src, ek, edge_v, entry_valid, Wk_list, Wv_list,
                      attn_cas_ch, attn_cfd_ch, conflict_bias=None):
        B, S, D = h_src.shape
        H, dh = self.heads, self.dh
        R = entry_valid.shape[-1]
        k_ch = torch.stack([m(h_src) for m in Wk_list], dim=2).view(B, S, R, H, dh)
        v_ch = torch.stack([m(h_src) for m in Wv_list], dim=2).view(B, S, R, H, dh)
        msg = v_ch + edge_v[:, :, None]
        s = self.leaky(q1[:, :, None] + k_ch + ek[:, :, None])            # [B,S,R,H,dh]
        a_cas = (s * attn_cas_ch[None, None]).sum(-1)                     # [B,S,R,H]
        a_cfd = (s * attn_cfd_ch[None, None]).sum(-1)
        if conflict_bias is not None:
            a_cas = a_cas - conflict_bias[:, :, None, None]
        neg_inf = torch.finfo(a_cas.dtype).min
        inv = ~entry_valid[..., None]                                     # [B,S,R,1]
        a_cas = a_cas.masked_fill(inv, neg_inf).view(B, S * R, H)
        a_cfd = a_cfd.masked_fill(inv, neg_inf).view(B, S * R, H)
        M_cas_h = torch.softmax(a_cas, dim=1)
        M_cfd_h = torch.softmax(a_cfd, dim=1)
        has_any = entry_valid.view(B, -1).any(dim=1)
        z = has_any[:, None, None].float()
        M_cas_h = torch.nan_to_num(M_cas_h) * z
        M_cfd_h = torch.nan_to_num(M_cfd_h) * z
        M_cas_h = M_cas_h.view(B, S, R, H).masked_fill(~entry_valid[..., None], 0.0)
        M_cfd_h = M_cfd_h.view(B, S, R, H).masked_fill(~entry_valid[..., None], 0.0)
        if getattr(self, 'uniform_mask', False):
            u = entry_valid[..., None].float().expand_as(M_cas_h)
            M_cas_h = u / u.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        cas = (M_cas_h[..., None] * msg).sum(dim=(1, 2)).reshape(B, self.dim)
        cfd = (M_cfd_h[..., None] * msg).sum(dim=(1, 2)).reshape(B, self.dim)
        M_cas_typed = M_cas_h.mean(-1)
        M_cfd_typed = M_cfd_h.mean(-1)
        src_cas = (M_cas_h[..., None] * msg).sum(dim=2).reshape(B, S, self.dim)   # [B,S,D]
        src_cfd = (M_cfd_h[..., None] * msg).sum(dim=2).reshape(B, S, self.dim)
        return (cas, cfd, M_cas_typed.sum(-1), M_cfd_typed.sum(-1), M_cas_typed, M_cfd_typed,
                src_cas, src_cfd)

    @staticmethod
    def _per_type(mods, x, types):
        stacked = torch.stack([m(x) for m in mods], dim=2)                  # [B,N,T,D]
        idx = types.clamp(min=0, max=len(mods) - 1)[:, :, None, None].expand(-1, -1, 1, x.shape[-1])
        return stacked.gather(2, idx).squeeze(2)                            # [B,N,D]

    def forward(self, h_ego, h_nbr, nbr_types, edge_ego, nbr_valid, h_map, edge_map, map_valid,
                conflict=None, ch_active=None, mch_active=None, ego_pred=None, h_ego_src=None):
        B, N = h_nbr.shape[0], h_nbr.shape[1]
        S = h_map.shape[1]
        H, dh = self.heads, self.dh

        q_ag = self.Wq_ag(h_ego).view(B, 1, H, dh)
        ek_ag = self.We_k_ag(edge_ego).view(B, N, H, dh)
        ev_ag = self.We_v_ag(edge_ego).view(B, N, H, dh)
        cb = None
        if self.conflict_bias and conflict is not None:
            cb = (F.softplus(self.conflict_w) * conflict[..., :3]).sum(-1)          # [B,N] >= 0
        M_cas_typed = M_cfd_typed = None
        agent_mass = map_mass = M_cas_raw = M_cas_map_raw = None
        ego_cas = ego_cfd = M_cas_ego_typed = M_cfd_ego_typed = ego_mass = None
        src_cas_ag = src_cas_mp = src_cfd_ag = src_cfd_mp = None
        joint = (self.joint_softmax and self.typed_kv
                 and ch_active is not None and mch_active is not None)
        if joint:
            fam_ag = self._typed_logits(q_ag, h_nbr, ek_ag, ev_ag,
                                        nbr_valid[:, :, None] & ch_active,
                                        self.Wk_ch, self.Wv_ch,
                                        self.attn_cas_ch, self.attn_cfd_ch, conflict_bias=cb)
            fam_ag = fam_ag + (nbr_valid[:, :, None] & ch_active,)
            ent_cas_mean = ent_cas_headmean = ent_cfd_mean = ent_cfd_headmean = \
                torch.zeros(B, device=h_ego.device)
        elif self.typed_kv and ch_active is not None:
            entry_valid = nbr_valid[:, :, None] & ch_active
            (ag_cas, ag_cfd, M_cas_ag, M_cfd_ag, M_cas_typed, M_cfd_typed,
             src_cas_ag, src_cfd_ag) = self._attend_typed(
                q_ag, h_nbr, ek_ag, ev_ag, entry_valid, self.Wk_ch, self.Wv_ch,
                self.attn_cas_ch, self.attn_cfd_ch, conflict_bias=cb)
            ent_cas_mean = ent_cas_headmean = ent_cfd_mean = ent_cfd_headmean = \
                torch.zeros(B, device=h_ego.device)
        else:
            k_ag = self._per_type(self.Wk_ag, h_nbr, nbr_types).view(B, N, H, dh)
            msg_ag = self._per_type(self.Wv_ag, h_nbr, nbr_types).view(B, N, H, dh) + ev_ag
            (ag_cas, ag_cfd, M_cas_ag, M_cfd_ag,
             ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean) = self._attend(
                q_ag, k_ag, ek_ag, msg_ag, nbr_valid, self.attn_cas, self.attn_cfd, self.gate_bias[0:2],
                conflict_bias=cb)

        q_mp = self.Wq_mp(h_ego).view(B, 1, H, dh)
        ek_mp = self.We_k_mp(edge_map).view(B, S, H, dh)
        ev_mp = self.We_v_mp(edge_map).view(B, S, H, dh)
        M_cas_mp_typed = M_cfd_mp_typed = None
        if joint:
            ev_mp_valid = map_valid[:, :, None] & mch_active
            fam_mp = self._typed_logits(q_mp, h_map, ek_mp, ev_mp, ev_mp_valid,
                                        self.Wk_mch, self.Wv_mch,
                                        self.attn_cas_mch, self.attn_cfd_mch) + (ev_mp_valid,)
            fam_ego = None
            if self.ego_family and ego_pred is not None and h_ego_src is not None:
                q_ego = self.Wq_ego(h_ego).view(B, 1, H, dh)
                ek_ego = self.ego_edge_k.view(1, 1, H, dh).expand(B, 1, H, dh)
                ev_ego = self.ego_edge_v.view(1, 1, H, dh).expand(B, 1, H, dh)
                fam_ego = self._typed_logits(q_ego, h_ego_src, ek_ego, ev_ego, ego_pred,
                                             self.Wk_ech, self.Wv_ech, self.attn_cas_ech, self.attn_cfd_ech) + (ego_pred,)
            outs = self._attend_typed_joint(*([fam_ag, fam_mp] + ([fam_ego] if fam_ego is not None else [])))
            ag_cas, ag_cfd, M_cas_typed, M_cfd_typed, src_cas_ag, src_cfd_ag = outs[0]
            mp_cas, mp_cfd, M_cas_mp_typed, M_cfd_mp_typed, src_cas_mp, src_cfd_mp = outs[1]
            if fam_ego is not None:
                ego_cas, ego_cfd, M_cas_ego_typed, M_cfd_ego_typed, _, _ = outs[2]
                ego_mass = M_cas_ego_typed.sum(dim=(1, 2))
            agent_mass = M_cas_typed.sum(dim=(1, 2))                       # [B]
            map_mass = M_cas_mp_typed.sum(dim=(1, 2))                      # [B]
            M_cas_raw = M_cas_typed.sum(-1)
            M_cas_ag = M_cas_typed.sum(-1) / agent_mass[:, None].clamp(min=1e-9)
            M_cfd_ag = M_cfd_typed.sum(-1) / M_cfd_typed.sum(dim=(1, 2))[:, None].clamp(min=1e-9)
            M_cas_map_raw = M_cas_mp_typed.sum(-1)
            M_cas_mp = M_cas_mp_typed.sum(-1) / map_mass[:, None].clamp(min=1e-9)
            M_cfd_mp = M_cfd_mp_typed.sum(-1) / M_cfd_mp_typed.sum(dim=(1, 2))[:, None].clamp(min=1e-9)
            ent_cas_mp_mean = ent_cas_mp_headmean = ent_cfd_mp_mean = ent_cfd_mp_headmean = \
                torch.zeros(B, device=h_ego.device)
        elif self.typed_kv and mch_active is not None:
            entry_valid_mp = map_valid[:, :, None] & mch_active
            (mp_cas, mp_cfd, M_cas_mp, M_cfd_mp, M_cas_mp_typed, M_cfd_mp_typed,
             src_cas_mp, src_cfd_mp) = self._attend_typed(
                q_mp, h_map, ek_mp, ev_mp, entry_valid_mp, self.Wk_mch, self.Wv_mch,
                self.attn_cas_mch, self.attn_cfd_mch)
            ent_cas_mp_mean = ent_cas_mp_headmean = ent_cfd_mp_mean = ent_cfd_mp_headmean = \
                torch.zeros(B, device=h_ego.device)
        else:
            k_mp = self.Wk_mp(h_map).view(B, S, H, dh)
            msg_mp = self.Wv_mp(h_map).view(B, S, H, dh) + ev_mp
            (mp_cas, mp_cfd, M_cas_mp, M_cfd_mp,
             ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean) = self._attend(
                q_mp, k_mp, ek_mp, msg_mp, map_valid, self.attn_cas_mp, self.attn_cfd_mp, self.gate_bias[2:4])

        if self.ego_family and not joint and ego_pred is not None and h_ego_src is not None:
            q_ego = self.Wq_ego(h_ego).view(B, 1, H, dh)
            ek_ego = self.ego_edge_k.view(1, 1, H, dh).expand(B, 1, H, dh)
            ev_ego = self.ego_edge_v.view(1, 1, H, dh).expand(B, 1, H, dh)
            (ego_cas, ego_cfd, _e1, _e2, M_cas_ego_typed, M_cfd_ego_typed, _e3, _e4) = self._attend_typed(
                q_ego, h_ego_src, ek_ego, ev_ego, ego_pred, self.Wk_ech, self.Wv_ech,
                self.attn_cas_ech, self.attn_cfd_ech)
            ego_mass = M_cas_ego_typed.sum(dim=(1, 2))

        if self.typed_kv and ch_active is not None:
            msg_ag = self._per_type(self.Wv_ag, h_nbr, nbr_types).view(B, N, H, dh) + ev_ag
        if self.typed_kv and mch_active is not None:
            msg_mp = self.Wv_mp(h_map).view(B, S, H, dh) + ev_mp
        w_ag = nbr_valid.float()
        w_ag = w_ag / w_ag.sum(-1, keepdim=True).clamp(min=1.0)               # [B,N]
        all_ag = (w_ag[:, :, None, None] * msg_ag).sum(dim=1).reshape(B, self.dim)
        w_mp = map_valid.float()
        w_mp = w_mp / w_mp.sum(-1, keepdim=True).clamp(min=1.0)               # [B,S]
        all_mp = (w_mp[:, :, None, None] * msg_mp).sum(dim=1).reshape(B, self.dim)

        self_fea = self.self_fc(h_ego)                                        # [B,D]
        f_all = torch.cat([self_fea, all_ag, all_mp], dim=-1)
        cat_cas, cat_cfd = [self_fea, ag_cas, mp_cas], [self_fea, ag_cfd, mp_cfd]
        if self.ego_family:
            cat_cas.append(ego_cas if ego_cas is not None else torch.zeros_like(self_fea))
            cat_cfd.append(ego_cfd if ego_cfd is not None else torch.zeros_like(self_fea))
        cas_pre = self.out_fc_cas(torch.cat(cat_cas, dim=-1))
        f_cas = self.norm_cas(cas_pre + h_ego if self.ego_residual else cas_pre)
        f_cfd = self.norm_cfd(self.out_fc_cfd(torch.cat(cat_cfd, dim=-1)))
        f_cas = self.ffn_cas(f_cas)
        f_cfd = self.ffn_cfd(f_cfd)
        f_cas = self.dropout(f_cas)
        f_cfd = self.dropout(f_cfd)
        with torch.no_grad():
            gate_cos = F.cosine_similarity(f_cas, h_ego, dim=-1)       # [B]
        h_ego_new = f_cas
        return (h_ego_new, f_cas, f_cfd, M_cas_ag, M_cfd_ag, M_cas_mp, M_cfd_mp,
                ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean,
                ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean,
                gate_cos, f_all, M_cas_typed, M_cfd_typed, M_cas_mp_typed, M_cfd_mp_typed,
                agent_mass, map_mass, M_cas_raw, M_cas_map_raw,
                src_cas_ag, src_cas_mp, src_cfd_ag, src_cfd_mp,
                M_cas_ego_typed, M_cfd_ego_typed, ego_mass)


class EgoCausalDisentangler(nn.Module):

    def __init__(self, dim=256, heads=8, layers=3, dropout=0.1, nbr_enrich=0, gate='softmax',
                 conflict_feats=0, conflict_bias=0, gate_channels=0, typed_kv=0,
                 channel_evidence=0, gate_trust='all', l1_drop_input=0, channel_set='v2', ego_family=0, ego_route=1):
        super().__init__()
        self.ego_family = bool(ego_family)
        self.n_ego = NUM_EGO_PREDS if ego_route else 4
        assert channel_set in CHANNEL_SETS, channel_set
        self.channel_set = channel_set
        self.n_ch, self.n_mch = CHANNEL_SETS[channel_set]
        if channel_set == 'v3':
            assert not channel_evidence, 'the v3 channel set has no evidence features'
            assert gate_trust == 'all' and not l1_drop_input, 'gate_trust/l1_drop_input are defined on v2 channel indices'
        self.gate_channels = bool(gate_channels)
        self.typed_kv = bool(typed_kv)
        self.channel_evidence = bool(channel_evidence)
        assert gate_trust in ('all', 'reliable')
        self.gate_trust = gate_trust
        self.l1_drop_input = bool(l1_drop_input)
        self.conflict_feats = conflict_feats
        self.conflict_bias = conflict_bias
        self.compute_conflict = bool(conflict_feats or conflict_bias)
        self.aligned_mode = 'straight'
        self.future_encoder = FutureEncoder()
        self.future_fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())
        self.type_embedding = nn.Embedding(NUM_AGENT_TYPES, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.lane_encoder = PolylineEncoder(3, dim)
        self.crosswalk_encoder = PolylineEncoder(3, dim)
        self.route_encoder = PolylineEncoder(3, dim)
        self.map_norm = nn.LayerNorm(dim)
        self.nbr_enrich = nn.ModuleList([
            _NbrMapEnrichLayer(dim, heads, EDGE_FEATURE_DIM, dropout) for _ in range(nbr_enrich)
        ])
        self.layers = nn.ModuleList([
            EgoCausalLayer(dim, heads, EDGE_FEATURE_DIM, dropout, gate=gate,
                           ag_edge_dim=(EDGE_FEATURE_DIM
                                        + (CONFLICT_FEATURE_DIM if conflict_feats else 0)
                                        + (NUM_EVIDENCE if channel_evidence else 0)),
                           map_edge_dim=(EDGE_FEATURE_DIM
                                         + (NUM_MAP_EVIDENCE if channel_evidence else 0)),
                           conflict_bias=bool(conflict_bias),
                           typed_kv=bool(typed_kv), n_ch=self.n_ch, n_mch=self.n_mch,
                           ego_family=bool(ego_family), n_ego=self.n_ego) for _ in range(layers)
        ])

    def forward(self, agent_feat, agent_valid, agent_pose, agent_types, inputs,
                neighbor_futures=None, neighbor_states=None, ref_path=None):
        B, Na, D = agent_feat.shape
        N = Na - 1

        if neighbor_futures is not None and neighbor_states is not None:
            fut_in = neighbor_futures[:, :N].unsqueeze(2)                              # [B,N,1,T,2]
            fut_emb = self.future_encoder(fut_in, neighbor_states[:, :N]).squeeze(2)   # [B,N,D]
            nbr_fused = self.future_fuse(torch.cat([agent_feat[:, 1:1 + N], fut_emb], dim=-1))
            agent_feat = torch.cat([agent_feat[:, :1], nbr_fused], dim=1)              # [B,Na,D]

        h = self.input_norm(agent_feat + self.type_embedding(agent_types))            # [B,Na,D]
        edge = build_edge_features(agent_pose)                                         # [B,Na,Na,De]
        edge_ego = edge[:, 0, 1:]

        ego_clean = h[:, 0]
        nbr_valid = agent_valid[:, 1:]                                                 # [B,N]
        nbr_types = agent_types[:, 1:]

        need_ch = self.gate_channels or self.typed_kv or self.channel_evidence
        ch_active = ch_evid = mch_active = mch_evid = None
        if need_ch:
            if "channel_active" in inputs:
                ch_active = inputs["channel_active"][:, :N].bool()
                ch_evid = inputs["channel_evidence"][:, :N].float()
                mch_active = inputs["map_channel_active"].bool()
                mch_evid = inputs["map_channel_evidence"].float()
                assert ch_active.shape[-1] == self.n_ch and mch_active.shape[-1] == self.n_mch, \
                    f'channel width {tuple(ch_active.shape[-1:])}/{tuple(mch_active.shape[-1:])} != model {self.channel_set} ({self.n_ch}/{self.n_mch}); check the loader channel_set'
            elif neighbor_futures is not None and ref_path is not None and self.channel_set == 'v3':
                for _k in ('lane_tl', 'intersections', 'stop_polygons'):
                    assert _k in inputs, f"inputs['{_k}'] is required for on-the-fly v3 channels"
                ego_p = inputs['ego_agent_past']
                ch_active = compute_agent_channels_v3(
                    inputs['neighbor_agents_past'][:, :N], ego_p, ref_path, inputs['route_lanes'],
                    inputs['map_crosswalks'], inputs['intersections'], inputs['stop_polygons'],
                    inputs['lane_tl'], inputs['map_lanes'], neighbor_futures=neighbor_futures[:, :N])
                mch_active = compute_map_channels_v3(
                    inputs['map_lanes'], inputs['map_crosswalks'], inputs['route_lanes'], ref_path,
                    inputs['lane_tl'], inputs['intersections'], inputs['stop_polygons'],
                    ego_v=ego_p[:, -1, 3:5].norm(dim=-1), ego_past=ego_p)
                ch_evid = mch_evid = None
            elif neighbor_futures is not None and ref_path is not None:
                ch_active, ch_evid = compute_channels(
                    inputs["neighbor_agents_past"][:, :N], inputs["ego_agent_past"],
                    neighbor_futures[:, :N], ref_path)
                mch_active, mch_evid = compute_map_channels(
                    inputs["map_lanes"], inputs["map_crosswalks"],
                    inputs["route_lanes"], ref_path)
            if self.l1_drop_input and ch_active is not None:
                ch_active = ch_active.clone()
                ch_active[..., list(L1_PROMOTED_CHANNELS)] = False

        lanes = inputs['map_lanes'][..., :3].float()
        cwalks = inputs['map_crosswalks'][..., :3].float()
        routes = inputs['route_lanes'][..., :3].float()
        h_map = torch.cat([self.lane_encoder(lanes), self.crosswalk_encoder(cwalks),
                           self.route_encoder(routes)], dim=1)                          # [B,S,D]
        h_map = self.map_norm(h_map)
        lane_pose, lane_v = _polyline_pose_and_valid(lanes)
        cw_pose, cw_v = _polyline_pose_and_valid(cwalks)
        rt_pose, rt_v = _polyline_pose_and_valid(routes)
        map_pose = torch.cat([lane_pose, cw_pose, rt_pose], dim=1)                      # [B,S,5]
        map_valid = torch.cat([lane_v, cw_v, rt_v], dim=1)                              # [B,S]
        map_edge_full = build_edge_features(torch.cat([agent_pose[:, 0:1], map_pose], dim=1))  # [B,1+S,1+S,De]
        edge_map = map_edge_full[:, 0, 1:]                                              # [B,S,De]

        h_ego = h[:, 0]                                                                # [B,D]
        h_nbr = h[:, 1:]

        if len(self.nbr_enrich) > 0:
            nbr_map_edge = build_edge_features(torch.cat([agent_pose[:, 1:], map_pose], dim=1))  # [B,N+S,N+S,De]
            edge_nbr_map = nbr_map_edge[:, :N, N:]
            for enrich in self.nbr_enrich:
                h_nbr = enrich(h_nbr, h_map, edge_nbr_map, map_valid)

        conf = None
        if self.compute_conflict and neighbor_futures is not None and neighbor_states is not None:
            route_pts = inputs['route_lanes'][..., :2].float().reshape(B, -1, 2)       # [B,P,2]
            route_v = route_pts.abs().sum(-1) > 1e-6                                    # [B,P]
            ego_speed = torch.norm(agent_pose[:, 0, 3:5], dim=-1)                       # [B]
            ref_xy = ref_v = None
            if ref_path is not None:
                _sel = select_ego_corridor(ref_path)                                    # [B]
                rp = ref_path[torch.arange(ref_path.shape[0], device=ref_path.device),
                              _sel][..., :2].float()                                    # [B,R,2]
                ref_v = rp.abs().sum(-1) > 1e-6
                ref_xy = rp
            conf = _conflict_features(neighbor_futures[:, :N], neighbor_states[:, :N],
                                      route_pts, route_v, ego_speed,
                                      ref_xy=ref_xy, ref_valid=ref_v,
                                      aligned_mode=self.aligned_mode)                    # [B,N,4]
            if self.conflict_feats:
                edge_ego = torch.cat([edge_ego, conf], dim=-1)                          # [B,N,De+4]

        if self.channel_evidence and ch_evid is not None:
            edge_ego = torch.cat([edge_ego, ch_evid], dim=-1)
            edge_map = torch.cat([edge_map, mch_evid], dim=-1)
        gated_valid, gated_map_valid = nbr_valid, map_valid
        if self.gate_channels and ch_active is not None:
            gate_src = ch_active
            if self.gate_trust == 'reliable':
                gate_src = ch_active.clone()
                gate_src[..., list(UNRELIABLE_CHANNELS)] = False
            gated_valid = nbr_valid & gate_src.any(-1)
            gated_map_valid = map_valid & mch_active.any(-1)

        f_cas = f_cfd = M_cas = M_cfd = M_cas_mp = M_cfd_mp = f_all = None
        ent_cas_mean = ent_cas_headmean = ent_cfd_mean = ent_cfd_headmean = None
        ent_cas_mp_mean = ent_cas_mp_headmean = ent_cfd_mp_mean = ent_cfd_mp_headmean = None
        M_cas_ty = M_cfd_ty = M_cas_mp_ty = M_cfd_mp_ty = None
        M_cas_ego_ty = M_cfd_ego_ty = ego_mass = None
        if self.ego_family:
            assert ref_path is not None, 'ego_family: ref_path is required for the route predicates'
        ego_pred = ego_predicates(inputs['ego_agent_past'], ref_path)[..., :self.n_ego] if self.ego_family else None
        gate_cos_layers = []
        for layer in self.layers:
            (h_ego, f_cas, f_cfd, M_cas, M_cfd, M_cas_mp, M_cfd_mp,
             ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean,
             ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean,
             gate_cos, f_all, M_cas_ty, M_cfd_ty, M_cas_mp_ty, M_cfd_mp_ty,
             agent_mass, map_mass, M_cas_raw, M_cas_map_raw,
             src_cas_ag, src_cas_mp, src_cfd_ag, src_cfd_mp,
             M_cas_ego_ty, M_cfd_ego_ty, ego_mass) = layer(
                h_ego, h_nbr, nbr_types, edge_ego, gated_valid, h_map, edge_map, gated_map_valid,
                conflict=conf,
                ch_active=(ch_active if self.typed_kv else None),
                mch_active=(mch_active if self.typed_kv else None),
                ego_pred=ego_pred, h_ego_src=(ego_clean[:, None] if self.ego_family else None))
            gate_cos_layers.append(gate_cos)
        gate_cos_stack = torch.stack(gate_cos_layers, dim=1)

        return {
            'f_cas': f_cas, 'f_cfd': f_cfd, 'M_cas': M_cas, 'M_cfd': M_cfd,
            'M_cas_map': M_cas_mp, 'M_cfd_map': M_cfd_mp, 'map_valid': map_valid,
            'agent_mass': agent_mass, 'map_mass': map_mass, 'M_cas_raw': M_cas_raw, 'M_cas_map_raw': M_cas_map_raw,
            'f_all': f_all,
            'conflict': conf,
            'M_cas_ent': ent_cas_mean, 'M_cas_headent': ent_cas_headmean,
            'M_cfd_ent': ent_cfd_mean, 'M_cfd_headent': ent_cfd_headmean,
            'M_cas_map_ent': ent_cas_mp_mean, 'M_cas_map_headent': ent_cas_mp_headmean,
            'M_cfd_map_ent': ent_cfd_mp_mean, 'M_cfd_map_headent': ent_cfd_mp_headmean,
            'ego_feat': h_ego, 'ego_clean': ego_clean, 'nbr_valid': nbr_valid,
            'M_cas_typed': M_cas_ty, 'M_cfd_typed': M_cfd_ty,
            'M_cas_ego_typed': M_cas_ego_ty, 'M_cfd_ego_typed': M_cfd_ego_ty, 'ego_mass': ego_mass, 'ego_pred': ego_pred,
            'M_cas_map_typed': M_cas_mp_ty, 'M_cfd_map_typed': M_cfd_mp_ty,
            'gated_valid': gated_valid, 'gated_map_valid': gated_map_valid,
            'ch_active': ch_active, 'mch_active': mch_active,
            'src_cas_ag': src_cas_ag, 'src_cas_mp': src_cas_mp,
            'src_cfd_ag': src_cfd_ag, 'src_cfd_mp': src_cfd_mp,
            'gate_cos': gate_cos_stack,
        }


class CausalEgoHead(nn.Module):

    def __init__(self, dim=256, modes=6, dropout=0.1, num_maneuvers=5,
                 dod_meta=False, num_lon=9, num_lat=7, dec_moe=False, lat_moe=False):
        super().__init__()
        self.dim, self.modes = dim, modes
        self.dod_meta = bool(dod_meta)
        self.dec_moe = bool(dec_moe)
        self.lat_moe = bool(lat_moe)
        assert not (self.dec_moe and self.lat_moe), "dec_moe cannot be combined with lat_moe"
        self.mode_query = nn.Embedding(modes, dim)
        if self.dod_meta:
            self.decision_emb_lon = nn.Embedding(num_lon, dim)
            self.decision_emb_lat = nn.Embedding(num_lat, dim)
        else:
            self.decision_emb = nn.Embedding(num_maneuvers, dim)
        self.cross = CrossTransformer(dim=dim, dropout=dropout)
        if self.dec_moe:
            assert self.dod_meta, "dec_moe requires dod_meta"
            assert num_lon == NUM_LON5 and num_lat == NUM_LAT5, \
                "dec_moe requires a 5x5 vocabulary (num_lon=5, num_lat=5)"
            self.register_buffer('lon_family', torch.tensor(LON5_FAMILY, dtype=torch.long))
            self.register_buffer('lat_family', torch.tensor(LAT5_FAMILY, dtype=torch.long))
            self.q_enh_fam = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                 for _ in range(NUM_FAMILIES)])
            self.predictor_fam = nn.ModuleList(
                [GMMPredictor(modalities=modes) for _ in range(NUM_FAMILIES)])
        elif self.lat_moe:
            assert self.dod_meta, "lat_moe requires dod_meta"
            self.q_enh = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.predictor_fam = nn.ModuleList(
                [GMMPredictor(modalities=modes) for _ in range(num_lat)])
        else:
            self.q_enh = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.predictor = GMMPredictor(modalities=modes)

    def forward(self, f_cas, ego_token, b_star=None):
        B = f_cas.shape[0]
        ctx = torch.cat([f_cas[:, None], ego_token[:, None]], dim=1)             # [B,2,D]
        ctx_pad = torch.zeros(B, 2, dtype=torch.bool, device=f_cas.device)
        q = self.mode_query.weight[None].expand(B, -1, -1)                       # [B,M,D]
        if self.dec_moe:
            assert b_star is not None, "dec_moe requires a decision"
            b_lon, b_lat = b_star
            dec_vec = self.decision_emb_lon(b_lon) + self.decision_emb_lat(b_lat)
            dec = dec_vec[:, None].expand(-1, self.modes, -1)                    # [B,M,D]
            qc = torch.cat([q, dec], dim=-1)                                     # [B,M,2D]
            qe = None
            famL = self.lat_family[b_lat]                                        # [B] 0/1/2
            for f in range(len(self.q_enh_fam)):
                idx = (famL == f).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                out_f = self.q_enh_fam[f](qc[idx])                               # [b,M,D]
                if qe is None:
                    qe = out_f.new_zeros(B, self.modes, self.dim)
                qe[idx] = out_f
            content = self.cross(qe, ctx, ctx, mask=ctx_pad)
            traj = score = None
            famS = self.lon_family[b_lon]                                        # [B] 0/1/2
            for f in range(len(self.predictor_fam)):
                idx = (famS == f).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                tf_, sf_ = self.predictor_fam[f](content[idx].unsqueeze(1))      # [b,1,M,80,4], [b,1,M]
                if traj is None:
                    traj = tf_.new_zeros((B,) + tf_.shape[1:])
                    score = sf_.new_zeros((B,) + sf_.shape[1:])
                traj[idx], score[idx] = tf_, sf_
            return traj, score
        if self.lat_moe:
            assert b_star is not None, "lat_moe requires a decision"
            b_lon, b_lat = b_star                                                # ([B], [B])
            dec_vec = self.decision_emb_lon(b_lon) + self.decision_emb_lat(b_lat)
            dec = dec_vec[:, None].expand(-1, self.modes, -1)                    # [B,M,D]
            q = self.q_enh(torch.cat([q, dec], dim=-1))
            content = self.cross(q, ctx, ctx, mask=ctx_pad)
            traj = score = None
            for c in range(len(self.predictor_fam)):
                idx = (b_lat == c).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                tf_, sf_ = self.predictor_fam[c](content[idx].unsqueeze(1))      # [b,1,M,80,4]
                if traj is None:
                    traj = tf_.new_zeros((B,) + tf_.shape[1:])
                    score = sf_.new_zeros((B,) + sf_.shape[1:])
                traj[idx], score[idx] = tf_, sf_
            return traj, score
        if b_star is not None:
            if self.dod_meta:
                b_lon, b_lat = b_star                                            # ([B], [B])
                dec_vec = self.decision_emb_lon(b_lon) + self.decision_emb_lat(b_lat)
            else:
                dec_vec = self.decision_emb(b_star)
            dec = dec_vec[:, None].expand(-1, self.modes, -1)                    # [B,M,D]
            q = self.q_enh(torch.cat([q, dec], dim=-1))
        content = self.cross(q, ctx, ctx, mask=ctx_pad)                          # [B,M,D]
        traj, score = self.predictor(content.unsqueeze(1))                       # [B,1,M,80,4], [B,1,M]
        return traj, score


class CausalPlanner(nn.Module):

    def __init__(self, dim=256, heads=8, layers=3, modes=6, dropout=0.1, nbr_enrich=0, num_maneuvers=5,
                 recon_drop=0.5, num_neighbors=10, future_steps=80, gate='softmax',
                 conflict_feats=0, conflict_bias=0, compute_conflict=0, aligned_mode='straight',
                 ego_residual=1, gate_channels=0, typed_kv=0, channel_evidence=0, gate_trust='all',
                 l1_drop_input=0,
                 dod_meta=0, num_lon=9, num_lat=7, uniform_mask=0, joint_softmax=0, dec_moe=0,
                 dod_tf=0, lat_moe=0, l1=0, l1_bottleneck=0,
                 num_l1_ag=6, num_l1_mp=2, channel_set='v2', psi_l0=0, psi_ego=0, ego_family=0, ego_route=1,
                 psi_linear=0):
        super().__init__()
        self.ego_family = bool(ego_family)
        self.n_ego = NUM_EGO_PREDS if ego_route else 4
        if self.ego_family:
            assert typed_kv, 'ego_family requires typed_kv'
        self.channel_set = channel_set
        self.n_ch, self.n_mch = CHANNEL_SETS[channel_set]
        self.psi_l0, self.psi_ego = bool(psi_l0), bool(psi_ego)
        if self.psi_l0:
            assert typed_kv, 'psi_l0 requires typed_kv (per-channel mass M_cas_typed)'
        self.dod_meta = bool(dod_meta)
        self.dec_moe = bool(dec_moe)
        self.lat_moe = int(lat_moe)
        self.dod_tf = bool(dod_tf)
        self.ss_p = 0.0
        self.psi_prior_alpha = 0.0
        if self.lat_moe:
            _w_lon = LON4_CE_WEIGHT
            _w_lat = LAT5L_CE_WEIGHT if int(lat_moe) >= 2 else LAT5V_CE_WEIGHT
        elif self.dec_moe:
            _w_lon, _w_lat = LON5_CE_WEIGHT, LAT5_CE_WEIGHT
        elif num_lon == NUM_LON_MERGED:
            _w_lon, _w_lat = LON_MERGED_CE_WEIGHT, LAT_CE_WEIGHT
        else:
            _w_lon, _w_lat = LON_CE_WEIGHT, LAT_CE_WEIGHT
        self._psi_log_w_lon = torch.log(torch.tensor(_w_lon, dtype=torch.float32))
        self._psi_log_w_lat = torch.log(torch.tensor(_w_lat, dtype=torch.float32))
        assert not (self.dod_tf and not self.dod_meta), "dod_tf requires dod_meta"
        assert not (self.dod_tf and self.dec_moe), "dod_tf cannot be combined with dec_moe (dec_moe already includes teacher forcing)"
        assert not (self.lat_moe and (self.dec_moe or self.dod_tf)), \
            "lat_moe must be used alone (it already includes teacher forcing)"
        assert not (self.lat_moe and not self.dod_meta), "lat_moe requires dod_meta"
        if self.dec_moe:
            self.register_buffer('lon5_remap', torch.tensor(LON5_MAP, dtype=torch.long))
            self.register_buffer('lat5_remap', torch.tensor(LAT5_MAP, dtype=torch.long))
        if self.lat_moe:
            _lat_map = LAT5L_MAP if int(lat_moe) >= 2 else LAT5V_MAP
            self.register_buffer('lon4_remap', torch.tensor(LON4_MAP, dtype=torch.long))
            self.register_buffer('lat5v_remap', torch.tensor(_lat_map, dtype=torch.long))
        self.disentangler = EgoCausalDisentangler(dim, heads, layers, dropout, nbr_enrich=nbr_enrich,
                                                  gate=gate, conflict_feats=conflict_feats,
                                                  conflict_bias=conflict_bias,
                                                  gate_channels=gate_channels, typed_kv=typed_kv,
                                                  channel_evidence=channel_evidence,
                                                  gate_trust=gate_trust,
                                                  l1_drop_input=l1_drop_input,
                                                  channel_set=channel_set, ego_family=ego_family, ego_route=ego_route)
        for _l in self.disentangler.layers:
            _l.ego_residual = bool(ego_residual)
            _l.uniform_mask = bool(uniform_mask)
            _l.joint_softmax = bool(joint_softmax)
            if joint_softmax:
                _l.family_bias = nn.Parameter(torch.zeros(2, device=_l.attn_cas.device))
        self.disentangler.compute_conflict = bool(compute_conflict or conflict_feats or conflict_bias)
        self.disentangler.aligned_mode = aligned_mode
        self.head = CausalEgoHead(dim, modes, dropout, num_maneuvers=num_maneuvers,
                                  dod_meta=dod_meta, num_lon=num_lon, num_lat=num_lat,
                                  dec_moe=dec_moe, lat_moe=lat_moe)
        self.l1 = bool(l1)
        self.l1_bottleneck = bool(l1_bottleneck)
        self.num_l1_ag, self.num_l1_mp = num_l1_ag, num_l1_mp
        psi_in = dim
        if self.l1:
            self.l1_head_ag = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, num_l1_ag))
            self.l1_head_mp = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, num_l1_mp))
            l1_sum = num_l1_ag + num_l1_mp
            psi_in = l1_sum if self.l1_bottleneck else dim + l1_sum
        if self.psi_l0:
            psi_in += self.n_ch + self.n_mch
        if self.psi_ego:
            psi_in += NUM_EGO_CONCEPTS
        if self.ego_family:
            psi_in += self.n_ego
        _psi = (lambda n_out: nn.Linear(psi_in, n_out)) if psi_linear else \
               (lambda n_out: nn.Sequential(nn.Linear(psi_in, 128), nn.ReLU(), nn.Linear(128, n_out)))
        self.psi_linear = bool(psi_linear)
        if self.dod_meta:
            self.psi_lon = _psi(num_lon)
            self.psi_lat = _psi(num_lat)
        else:
            self.psi = _psi(num_maneuvers)

        self.recon_drop = recon_drop
        self.cfd_recon = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, 3 * dim))

        self.num_neighbors, self.future_steps = num_neighbors, future_steps
        self.nbr_head = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(),
                                      nn.Linear(2 * dim, num_neighbors * future_steps * 2))

    def forward(self, encoder_outputs, inputs, num_agents,
                neighbor_futures=None, neighbor_states=None, also_cfd_plan=False, ref_path=None):
        Na = num_agents
        agent_feat = encoder_outputs['agent_tokens'][:, :Na].detach()
        agent_valid = ~encoder_outputs['mask'][:, :Na]
        agent_pose = encoder_outputs['actors'][:, :Na, -1].detach()     # [B,Na,5]
        agent_types = _agent_types(inputs['neighbor_agents_past'], Na - 1)

        dis = self.disentangler(agent_feat, agent_valid, agent_pose, agent_types, inputs,
                                neighbor_futures=neighbor_futures, neighbor_states=neighbor_states,
                                ref_path=ref_path)
        f_cas, f_cfd = dis['f_cas'], dis['f_cfd']

        keep = 1.0
        if self.training and self.recon_drop > 0.0:
            keep = (torch.rand(f_cas.shape[0], 1, device=f_cas.device) >= self.recon_drop).float()
        recon_pred = self.cfd_recon(torch.cat([keep * f_cas, f_cfd], dim=-1))   # [B,3D]

        nbr_pred = self.nbr_head(f_cfd).view(-1, self.num_neighbors, self.future_steps, 2)

        psi_cas = psi_cfd = None
        psi_lon_cas = psi_lat_cas = psi_lon_cfd = psi_lat_cfd = None
        l1_ag = l1_mp = None
        z_cas, z_cfd = f_cas, f_cfd
        if self.l1 and dis['src_cas_ag'] is not None:
            l1_ag = self.l1_head_ag(dis['src_cas_ag'])                  # [B,N,C_ag]
            l1_mp = self.l1_head_mp(dis['src_cas_mp'])                  # [B,S,C_mp]
            joint = dis.get('M_cas_raw') is not None
            w_ag = dis['M_cas_raw'] if joint else dis['M_cas']
            w_mp = dis['M_cas_map_raw'] if joint else dis['M_cas_map']
            va = (dis['gated_valid'].float() * w_ag)[..., None]
            vm = (dis['gated_map_valid'].float() * w_mp)[..., None]
            summ = torch.cat([(torch.softmax(l1_ag, -1) * va).sum(1),
                              (torch.softmax(l1_mp, -1) * vm).sum(1)], dim=-1)   # [B,C_ag+C_mp]
            z_cas = summ if self.l1_bottleneck else torch.cat([f_cas, summ], dim=-1)
            wc_ag = dis['M_cfd_typed'].sum(-1) if joint else dis['M_cfd']
            wc_mp = dis['M_cfd_map_typed'].sum(-1) if joint else dis['M_cfd_map']
            vac = (dis['gated_valid'].float() * wc_ag)[..., None]
            vmc = (dis['gated_map_valid'].float() * wc_mp)[..., None]
            summ_cfd = torch.cat([(torch.softmax(self.l1_head_ag(dis['src_cfd_ag']), -1) * vac).sum(1),
                                  (torch.softmax(self.l1_head_mp(dis['src_cfd_mp']), -1) * vmc).sum(1)],
                                 dim=-1)
            z_cfd = summ_cfd if self.l1_bottleneck else torch.cat([f_cfd, summ_cfd], dim=-1)
        summ_l0_ag = summ_l0_mp = ego_concepts = None
        extra_cas, extra_cfd = [], []
        if self.psi_l0:
            assert dis['M_cas_typed'] is not None, 'psi_l0: M_cas_typed missing (is typed_kv off?)'
            summ_l0_ag = dis['M_cas_typed'].sum(1)
            summ_l0_mp = dis['M_cas_map_typed'].sum(1)                # [B,R_mp]
            extra_cas += [summ_l0_ag, summ_l0_mp]
            extra_cfd += [dis['M_cfd_typed'].sum(1), dis['M_cfd_map_typed'].sum(1)]
        if self.psi_ego:
            assert 'intersections' in inputs and ref_path is not None, 'psi_ego: inputs[intersections] and ref_path are required (entersJunction)'
            ego_concepts = ego_concept_features(inputs['ego_agent_past'], ref_path, inputs['intersections'])   # [B,5]
            extra_cas.append(ego_concepts)
            extra_cfd.append(torch.zeros_like(ego_concepts))
        summ_ego = None
        if self.ego_family:
            assert dis['M_cas_ego_typed'] is not None, 'ego_family: ego mass missing (joint_softmax/typed_kv?)'
            summ_ego = dis['M_cas_ego_typed'].sum(1)
            extra_cas.append(summ_ego)
            extra_cfd.append(dis['M_cfd_ego_typed'].sum(1))
        if extra_cas:
            z_cas = torch.cat([z_cas] + extra_cas, dim=-1)
            z_cfd = torch.cat([z_cfd] + extra_cfd, dim=-1)
        if self.dod_meta:
            psi_lon_cas, psi_lat_cas = self.psi_lon(z_cas), self.psi_lat(z_cas)
            psi_lon_cfd, psi_lat_cfd = self.psi_lon(z_cfd), self.psi_lat(z_cfd)
            if self.psi_prior_alpha and not self.training:
                _a = float(self.psi_prior_alpha)
                _wl = self._psi_log_w_lon.to(psi_lon_cas.device)
                _wt = self._psi_log_w_lat.to(psi_lat_cas.device)
                psi_lon_cas, psi_lat_cas = psi_lon_cas - _a * _wl, psi_lat_cas - _a * _wt
                psi_lon_cfd, psi_lat_cfd = psi_lon_cfd - _a * _wl, psi_lat_cfd - _a * _wt
            b_star = (psi_lon_cas.argmax(-1), psi_lat_cas.argmax(-1))
            b_cfd = (psi_lon_cfd.argmax(-1), psi_lat_cfd.argmax(-1))
        else:
            psi_cas = self.psi(z_cas)
            psi_cfd = self.psi(z_cfd)
            b_star = psi_cas.argmax(-1)
            b_cfd = psi_cfd.argmax(-1)

        b_head = b_star
        if (self.dec_moe or self.dod_tf or self.lat_moe) and self.training:
            gt_lon, gt_lat = inputs.get('decision_lon'), inputs.get('decision_lat')
            if gt_lon is not None:
                if self.dec_moe:
                    tl, tt = self.lon5_remap[gt_lon], self.lat5_remap[gt_lat]
                elif self.lat_moe:
                    tl, tt = self.lon4_remap[gt_lon], self.lat5v_remap[gt_lat]
                else:
                    tl, tt = gt_lon, gt_lat
                if self.ss_p > 0.0:
                    u = torch.rand(tl.shape[0], device=tl.device) < self.ss_p
                    tl = torch.where(u, b_star[0], tl)
                    tt = torch.where(u, b_star[1], tt)
                b_head = (tl, tt)
        traj, score = self.head(f_cas, dis['ego_clean'], b_head)        # [B,1,M,80,4], [B,1,M]

        traj_cfd = score_cfd = None
        if also_cfd_plan:
            traj_cfd, score_cfd = self.head(f_cfd, dis['ego_clean'], b_cfd)

        out = {
            'traj': traj, 'score': score,
            'traj_cfd': traj_cfd, 'score_cfd': score_cfd,
            'M_cas': dis['M_cas'], 'M_cfd': dis['M_cfd'],
            'M_cas_map': dis['M_cas_map'], 'M_cfd_map': dis['M_cfd_map'],
            'agent_mass': dis['agent_mass'], 'map_mass': dis['map_mass'],
            'M_cas_raw': dis['M_cas_raw'], 'M_cas_map_raw': dis['M_cas_map_raw'],
            'map_valid': dis['map_valid'],
            'M_cas_ent': dis['M_cas_ent'], 'M_cas_headent': dis['M_cas_headent'],
            'M_cfd_ent': dis['M_cfd_ent'], 'M_cfd_headent': dis['M_cfd_headent'],
            'M_cas_map_ent': dis['M_cas_map_ent'], 'M_cas_map_headent': dis['M_cas_map_headent'],
            'M_cfd_map_ent': dis['M_cfd_map_ent'], 'M_cfd_map_headent': dis['M_cfd_map_headent'],
            'f_cas': f_cas, 'f_cfd': f_cfd,
            'ego_clean': dis['ego_clean'],
            'nbr_valid': dis['nbr_valid'],
            'gate_cos': dis['gate_cos'],
            'psi_cas': psi_cas,
            'psi_cfd': psi_cfd,
            'psi_lon_cas': psi_lon_cas, 'psi_lat_cas': psi_lat_cas,
            'psi_lon_cfd': psi_lon_cfd, 'psi_lat_cfd': psi_lat_cfd,
            'recon_pred': recon_pred,
            'f_all': dis['f_all'],
            'nbr_pred': nbr_pred,
            'conflict': dis['conflict'],
            'M_cas_typed': dis['M_cas_typed'], 'M_cfd_typed': dis['M_cfd_typed'],
            'M_cas_map_typed': dis['M_cas_map_typed'], 'M_cfd_map_typed': dis['M_cfd_map_typed'],
            'gated_valid': dis['gated_valid'], 'gated_map_valid': dis['gated_map_valid'],
            'ch_active': dis['ch_active'], 'mch_active': dis['mch_active'],
            'l1_ag': l1_ag, 'l1_mp': l1_mp,
            'summ_l0_ag': summ_l0_ag, 'summ_l0_mp': summ_l0_mp, 'ego_concepts': ego_concepts,
            'summ_ego': summ_ego, 'ego_mass': dis['ego_mass'], 'ego_pred': dis['ego_pred'],
        }
        return out

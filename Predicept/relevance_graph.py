
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_modules import FutureEncoder


NODE_TYPE_EGO = 0
NODE_TYPE_VEHICLE = 1
NODE_TYPE_PEDESTRIAN = 2
NODE_TYPE_BICYCLE = 3
NODE_TYPE_LANE = 4
NODE_TYPE_CROSSWALK = 5
NODE_TYPE_ROUTE = 6
NUM_NODE_TYPES = 7

EDGE_FEATURE_DIM = 7


class PolylineEncoder(nn.Module):

    def __init__(self, in_dim, dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        # x: [B, E, P, F]
        h = self.mlp(x)            # [B, E, P, D]
        h = torch.max(h, dim=2)[0]  # [B, E, D]
        return h


def _polyline_pose_and_valid(poly):
    B, E, P, _ = poly.shape
    pt_valid = (poly[..., :2].abs().sum(-1) > 1e-6)          # [B, E, P]
    elem_valid = pt_valid.any(dim=-1)                         # [B, E]

    cnt = pt_valid.sum(-1).clamp(min=1).unsqueeze(-1).float()  # [B, E, 1]
    m = pt_valid.unsqueeze(-1).float()                         # [B, E, P, 1]
    centroid = (poly[..., :2] * m).sum(2) / cnt                # [B, E, 2]

    heading = poly[..., 2]
    cos_m = (torch.cos(heading) * pt_valid.float()).sum(-1) / cnt.squeeze(-1)
    sin_m = (torch.sin(heading) * pt_valid.float()).sum(-1) / cnt.squeeze(-1)
    head = torch.atan2(sin_m, cos_m)                           # [B, E]

    zeros = torch.zeros(B, E, 2, device=poly.device, dtype=poly.dtype)
    pose = torch.cat([centroid, head.unsqueeze(-1), zeros], dim=-1)  # [B, E, 5]
    return pose, elem_valid


def build_edge_features(node_pose):
    p = node_pose
    xi = p[:, :, None, :]
    xj = p[:, None, :, :]
    dxy = xj[..., :2] - xi[..., :2]                       # [B,V,V,2]
    dist = torch.norm(dxy, dim=-1)                        # [B,V,V]
    dtheta = xj[..., 2] - xi[..., 2]                      # [B,V,V]
    dvxy = xj[..., 3:5] - xi[..., 3:5]                    # [B,V,V,2]
    edge = torch.stack([
        dxy[..., 0], dxy[..., 1], torch.log1p(dist),
        torch.cos(dtheta), torch.sin(dtheta),
        dvxy[..., 0], dvxy[..., 1],
    ], dim=-1)                                            # [B,V,V,7]
    return edge


class HeteroEdgeGATv2Layer(nn.Module):

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.We_k = nn.Linear(edge_dim, dim)
        self.We_v = nn.Linear(edge_dim, dim)
        self.attn_vec = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.attn_vec)

        self.out_proj = nn.Linear(dim, dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, edge, edge_allow):
        B, V, _ = h.shape
        H, dh = self.heads, self.dh

        q = self.Wq(h).view(B, V, H, dh)            # [B,V,H,dh]
        k = self.Wk(h).view(B, V, H, dh)
        v = self.Wv(h).view(B, V, H, dh)
        ek = self.We_k(edge).view(B, V, V, H, dh)   # [B,V,V,H,dh]
        ev = self.We_v(edge).view(B, V, V, H, dh)

        s = self.leaky_relu(q[:, :, None] + k[:, None, :] + ek)   # [B,V,V,H,dh]
        scores = (s * self.attn_vec).sum(-1)                       # [B,V,V,H]

        neg_inf = torch.finfo(scores.dtype).min
        mask = edge_allow.unsqueeze(-1)                            # [B,V,V,1]
        scores = scores.masked_fill(~mask, neg_inf)

        attn_full = torch.softmax(scores, dim=2)
        attn = self.dropout(attn_full)

        msg = v[:, None, :] + ev                                   # [B,V,V,H,dh]
        out = (attn.unsqueeze(-1) * msg).sum(dim=2)                # [B,V,H,dh]
        out = out.reshape(B, V, self.dim)
        out = self.out_proj(out)

        h = self.norm_1(h + self.dropout(out))
        h = self.norm_2(h + self.ffn(h))
        return h, attn_full.mean(dim=-1)                           # h, [B,V,V]


class SceneRelevanceGraph(nn.Module):

    def __init__(self, dim=256, heads=8, layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim

        self.lane_encoder = PolylineEncoder(3, dim)
        self.crosswalk_encoder = PolylineEncoder(3, dim)
        self.route_encoder = PolylineEncoder(3, dim)

        self.type_embedding = nn.Embedding(NUM_NODE_TYPES, dim)
        self.input_norm = nn.LayerNorm(dim)

        self.future_encoder = FutureEncoder()
        self.future_fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())

        self.layers = nn.ModuleList([
            HeteroEdgeGATv2Layer(dim, heads, EDGE_FEATURE_DIM, dropout) for _ in range(layers)
        ])


    def _agent_types(self, neighbor_agents_past, num_neighbors):
        B = neighbor_agents_past.shape[0]
        device = neighbor_agents_past.device
        nbr = neighbor_agents_past[:, :num_neighbors, -1]        # [B, N, F]
        if nbr.shape[-1] >= 11:
            onehot = nbr[..., 8:11]                              # [B, N, 3]
            cls = onehot.argmax(-1)
        else:
            cls = torch.zeros(B, num_neighbors, dtype=torch.long, device=device)
        nbr_type = cls + NODE_TYPE_VEHICLE                       # 1..3
        ego_type = torch.full((B, 1), NODE_TYPE_EGO, dtype=torch.long, device=device)
        return torch.cat([ego_type, nbr_type], dim=1)            # [B, 1+N]

    def forward(self, encoder_outputs, inputs, num_agents, return_attention=False,
                neighbor_futures=None, neighbor_states=None):
        device = (encoder_outputs['agent_tokens'] if 'agent_tokens' in encoder_outputs
                  else encoder_outputs['encoding']).device
        Na = num_agents

        if 'agent_tokens' in encoder_outputs:
            agent_feat = encoder_outputs['agent_tokens'][:, :Na].detach()
        else:
            agent_feat = encoder_outputs['encoding'][:, :Na].detach()
        agent_valid = ~encoder_outputs['mask'][:, :Na]
        agent_pose = encoder_outputs['actors'][:, :Na, -1].detach()        # [B, Na, 5]
        agent_types = self._agent_types(inputs['neighbor_agents_past'], Na - 1)  # [B, Na]
        B = agent_feat.shape[0]

        if neighbor_futures is not None and neighbor_states is not None:
            N = Na - 1
            fut_in = neighbor_futures[:, :N].unsqueeze(2)                              # [B, N, 1, T, 2]
            fut_emb = self.future_encoder(fut_in, neighbor_states[:, :N]).squeeze(2)   # [B, N, D]
            nbr_fused = self.future_fuse(torch.cat([agent_feat[:, 1:1 + N], fut_emb], dim=-1))  # [B, N, D]
            agent_feat = torch.cat([agent_feat[:, :1], nbr_fused], dim=1)              # [B, Na, D]

        lanes = inputs['map_lanes'][..., :3].float()
        cwalks = inputs['map_crosswalks'][..., :3].float()
        routes = inputs['route_lanes'][..., :3].float()

        lane_feat = self.lane_encoder(lanes)                               # [B, L, D]
        cw_feat = self.crosswalk_encoder(cwalks)                           # [B, C, D]
        rt_feat = self.route_encoder(routes)                               # [B, R, D]

        lane_pose, lane_valid = _polyline_pose_and_valid(lanes)
        cw_pose, cw_valid = _polyline_pose_and_valid(cwalks)
        rt_pose, rt_valid = _polyline_pose_and_valid(routes)

        L, C, R = lane_feat.shape[1], cw_feat.shape[1], rt_feat.shape[1]

        def _types(n, t):
            return torch.full((B, n), t, dtype=torch.long, device=device)

        feat = torch.cat([agent_feat, lane_feat, cw_feat, rt_feat], dim=1)     # [B, V, D]
        pose = torch.cat([agent_pose, lane_pose, cw_pose, rt_pose], dim=1)     # [B, V, 5]
        valid = torch.cat([agent_valid, lane_valid, cw_valid, rt_valid], dim=1)  # [B, V]
        ntypes = torch.cat([
            agent_types,
            _types(L, NODE_TYPE_LANE), _types(C, NODE_TYPE_CROSSWALK), _types(R, NODE_TYPE_ROUTE),
        ], dim=1)                                                              # [B, V]
        V = feat.shape[1]

        slices = {
            'agent': (0, Na), 'lane': (Na, Na + L),
            'crosswalk': (Na + L, Na + L + C), 'route': (Na + L + C, V),
        }

        h = self.input_norm(feat + self.type_embedding(ntypes))

        edge = build_edge_features(pose)                                       # [B,V,V,De]
        is_agent = (ntypes < NODE_TYPE_LANE)                                   # [B,V]
        touch_agent = is_agent[:, :, None] | is_agent[:, None, :]              # [B,V,V]
        edge_allow = valid[:, None, :] & touch_agent                          # [B,V,V]

        attns = []
        for layer in self.layers:
            h, a = layer(h, edge, edge_allow)
            attns.append(a)

        last_attn = attns[-1]
        importance = last_attn[:, 0, :]

        out = {
            'context': h,
            'valid': valid,
            'importance': importance,
            'node_types': ntypes,
            'node_pose': pose,
            'slices': slices,
        }
        if return_attention:
            out['attention'] = last_attn.detach()
        return out


_TYPE_NAMES = {
    NODE_TYPE_EGO: 'ego', NODE_TYPE_VEHICLE: 'vehicle', NODE_TYPE_PEDESTRIAN: 'pedestrian',
    NODE_TYPE_BICYCLE: 'bicycle', NODE_TYPE_LANE: 'lane', NODE_TYPE_CROSSWALK: 'crosswalk',
    NODE_TYPE_ROUTE: 'route',
}


def plot_scene_importance(graph_out, raw_inputs=None, batch_idx=0, save_path=None,
                          ax=None, title=None, topk=None):
    import numpy as np
    import matplotlib.pyplot as plt

    pose = graph_out['node_pose'][batch_idx].detach().cpu().numpy()     # [V, 5]
    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()     # [V]
    valid = graph_out['valid'][batch_idx].detach().cpu().numpy()        # [V]
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()  # [V]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    if raw_inputs is not None:
        def _np(x):
            return x[batch_idx].detach().cpu().numpy() if torch.is_tensor(x) else x[batch_idx]
        for key, style in (('map_lanes', '-'), ('route_lanes', '-'), ('map_crosswalks', '--')):
            if key in raw_inputs:
                polys = _np(raw_inputs[key])
                for poly in polys:
                    m = np.abs(poly[:, :2]).sum(-1) > 1e-6
                    if m.any():
                        ax.plot(poly[m, 0], poly[m, 1], style, color='lightgray', linewidth=0.8, zorder=0)

    highlight = np.ones_like(valid, dtype=bool)
    if topk is not None:
        order = np.argsort(-np.where(valid, imp, -np.inf))
        highlight = np.zeros_like(valid, dtype=bool)
        highlight[order[:topk]] = True

    vmax = max(imp[valid].max(), 1e-6) if valid.any() else 1.0
    sc = None
    for v in range(len(imp)):
        if not valid[v]:
            continue
        x, y = pose[v, 0], pose[v, 1]
        is_ego = (ntypes[v] == NODE_TYPE_EGO)
        is_agent = (ntypes[v] < NODE_TYPE_LANE)
        alpha = 1.0 if highlight[v] else 0.15
        if is_ego:
            ax.scatter(x, y, c='blue', s=200, marker='*', edgecolors='k',
                       zorder=6, label='ego')
            continue
        marker = 'o' if is_agent else 's'
        size = 60 + 600 * (imp[v] / vmax)
        sc = ax.scatter(x, y, c=[imp[v]], cmap='hot', vmin=0, vmax=vmax,
                        s=size, marker=marker, edgecolors='k', linewidths=0.6,
                        alpha=alpha, zorder=5)
        if highlight[v] and imp[v] > 0.3 * vmax:
            ax.annotate(_TYPE_NAMES.get(int(ntypes[v]), '?'), (x, y),
                        fontsize=7, zorder=7)

    if sc is not None:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='importance')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, ego frame)')
    ax.set_ylabel('y (m, ego frame)')
    ax.set_title(title or 'Scene relevance (agents = circles, map = squares)')
    ax.legend(loc='upper right', fontsize=8)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    return ax


def select_relevant_nodes(graph_out, top_k=5, batch_idx=0):
    import numpy as np
    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()
    valid = graph_out['valid'][batch_idx].detach().cpu().numpy()
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()
    ego_sel = np.where(valid & (ntypes == NODE_TYPE_EGO))[0]
    nonego = np.where(valid & (ntypes != NODE_TYPE_EGO))[0]
    top_idx = nonego[np.argsort(-imp[nonego])][:top_k] if len(nonego) else np.array([], dtype=int)
    return np.concatenate([ego_sel, top_idx]).astype(int)


def annotate_map_node_ids(ax, graph_out, shown, batch_idx=0, fontsize=6):
    import numpy as np
    pose = graph_out['node_pose'][batch_idx].detach().cpu().numpy()
    for v in np.asarray(shown).astype(int):
        ax.annotate(str(int(v)), (pose[v, 0], pose[v, 1]), fontsize=fontsize, color='red',
                    zorder=20, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='red', alpha=0.75, lw=0.4))


def plot_scene_graph(graph_out, raw_inputs=None, batch_idx=0, ax=None, save_path=None,
                     title=None, top_k=5, min_edge=0.01, annotate=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()   # [V]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    shown = select_relevant_nodes(graph_out, top_k=top_k, batch_idx=batch_idx)

    ego_nodes = [int(v) for v in shown if ntypes[v] == NODE_TYPE_EGO]
    others = [int(v) for v in shown if ntypes[v] != NODE_TYPE_EGO]
    layout = {}
    if ego_nodes:
        layout[ego_nodes[0]] = (0.0, 0.0)
    for k, node in enumerate(others):
        ang = np.pi / 2 - 2 * np.pi * k / max(len(others), 1)
        layout[node] = (float(np.cos(ang)), float(np.sin(ang)))

    others_imp = np.array([imp[v] for v in others], dtype=float)
    others_imp = others_imp / max(others_imp.sum(), 1e-9)
    imp_of = {node: float(w) for node, w in zip(others, others_imp)}

    ex, ey = layout[ego_nodes[0]] if ego_nodes else (0.0, 0.0)
    wmax = max(others_imp.max(), 1e-6) if len(others_imp) else 1.0
    for node in others:
        w = imp_of[node]
        if w < min_edge:
            continue
        x, y = layout[node]
        rel = w / wmax
        ax.plot([ex, x], [ey, y], '-', color='crimson',
                linewidth=1.0 + 6.0 * rel, alpha=0.35 + 0.6 * rel, zorder=2)
        mx, my = (ex + x) / 2, (ey + y) / 2
        ax.annotate(f"{w:.2f}", (mx, my), fontsize=8, color='darkred', zorder=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='crimson', alpha=0.9, lw=0.5))

    type_marker = {
        NODE_TYPE_VEHICLE: 'o', NODE_TYPE_PEDESTRIAN: 'P', NODE_TYPE_BICYCLE: 'X',
        NODE_TYPE_LANE: 's', NODE_TYPE_CROSSWALK: 'D', NODE_TYPE_ROUTE: '^',
    }
    vmax = max(others_imp.max(), 1e-6) if len(others_imp) else 1.0
    sc = None
    for v in shown:
        x, y = layout[int(v)]
        if ntypes[v] == NODE_TYPE_EGO:
            ax.scatter(x, y, c='deepskyblue', s=900, marker='*', edgecolors='k', linewidths=1.2, zorder=6)
        else:
            mk = type_marker.get(int(ntypes[v]), 'o')
            sc = ax.scatter(x, y, c=[imp_of[int(v)]], cmap='YlOrRd', vmin=0, vmax=vmax,
                            s=850, marker=mk, edgecolors='k', linewidths=1.0, zorder=5)
        if annotate:
            ax.annotate(str(int(v)), (x, y), fontsize=10, fontweight='bold', color='black',
                        zorder=9, ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec='none', alpha=0.6))

    if sc is not None:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                     label='importance (light yellow = low, dark red = high)')

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='deepskyblue', markeredgecolor='k', markersize=16, label='ego'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='vehicle'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='pedestrian'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='bicycle'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='lane'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='crosswalk'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='route'),
        Line2D([0], [0], color='crimson', lw=4, label='edge = importance to the ego (sums to 1)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title or 'Ego-star importance graph (edge = importance, sums to 1; node label = id)')

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    return shown


def build_relevance_record(features, graph_out, batch_idx=0, extra=None):
    import numpy as np

    def npy(x):
        return x[batch_idx].detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x[batch_idx])

    rec = {
        'importance': graph_out['importance'][batch_idx].detach().cpu().numpy().astype('float32'),
        'valid': graph_out['valid'][batch_idx].detach().cpu().numpy(),
        'node_types': graph_out['node_types'][batch_idx].detach().cpu().numpy().astype('int16'),
        'node_pose': graph_out['node_pose'][batch_idx].detach().cpu().numpy().astype('float32'),
        'slices': {k: tuple(int(i) for i in v) for k, v in graph_out['slices'].items()},
        'lanes_xy': npy(features['map_lanes'])[..., :2].astype('float32'),
        'route_xy': npy(features['route_lanes'])[..., :2].astype('float32'),
        'cw_xy': npy(features['map_crosswalks'])[..., :2].astype('float32'),
    }
    if 'attention' in graph_out:
        rec['attention'] = graph_out['attention'][batch_idx].detach().cpu().numpy().astype('float16')
    if extra:
        rec.update(extra)
    return rec


def draw_relevance(ax, rec, threshold=0.65, draw_base=True, annotate=True):
    import numpy as np

    imp = rec['importance']; valid = rec['valid']; nt = rec['node_types']
    pose = rec['node_pose']; sl = rec['slices']
    vmax = max(imp[valid].max(), 1e-6) if valid.any() else 1.0
    norm = imp / vmax
    passed = valid & (norm >= threshold)

    if draw_base:
        for arr in (rec['lanes_xy'], rec['route_xy'], rec['cw_xy']):
            for poly in arr:
                m = np.abs(poly).sum(-1) > 1e-6
                if m.any():
                    ax.plot(poly[m, 0], poly[m, 1], '-', color='lightgray', linewidth=0.7, zorder=0)

    def _poly_for(node):
        for key, arr_key in (('lane', 'lanes_xy'), ('crosswalk', 'cw_xy'), ('route', 'route_xy')):
            if key in sl and sl[key][0] <= node < sl[key][1]:
                return rec[arr_key][node - sl[key][0]]
        return None

    type_marker = {NODE_TYPE_VEHICLE: 'o', NODE_TYPE_PEDESTRIAN: 'P', NODE_TYPE_BICYCLE: 'X'}

    for v in np.where(passed)[0]:
        if nt[v] == NODE_TYPE_EGO:
            continue
        a0, a1 = sl['agent']
        if a0 <= v < a1:
            mk = type_marker.get(int(nt[v]), 'o')
            ax.scatter(pose[v, 0], pose[v, 1], s=320, marker=mk, facecolors='none',
                       edgecolors='red', linewidths=2.5, zorder=22)
            if annotate:
                ax.annotate(f"{norm[v]:.2f}", (pose[v, 0], pose[v, 1]), fontsize=8, color='red',
                            zorder=23, ha='center', va='bottom')
        else:
            poly = _poly_for(int(v))
            if poly is not None:
                m = np.abs(poly).sum(-1) > 1e-6
                if m.any():
                    ax.plot(poly[m, 0], poly[m, 1], '-', color='red', linewidth=3.0, alpha=0.9, zorder=10)
                    if annotate:
                        mid = poly[m][len(poly[m]) // 2]
                        ax.annotate(f"{norm[v]:.2f}", (mid[0], mid[1]), fontsize=7, color='darkred', zorder=11)

    a0 = sl['agent'][0]
    ax.scatter(pose[a0, 0], pose[a0, 1], c='deepskyblue', s=220, marker='*', edgecolors='k', zorder=25)
    return int(passed.sum())


def plot_bev_relevance(features, graph_out, ax=None, batch_idx=0, save_path=None,
                       title=None, top_k_each=3, cmap='YlOrRd'):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()
    valid = graph_out['valid'][batch_idx].detach().cpu().numpy()
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()
    pose = graph_out['node_pose'][batch_idx].detach().cpu().numpy()
    sl = graph_out['slices']

    def _np(x):
        return x[batch_idx].detach().cpu().numpy() if torch.is_tensor(x) else x[batch_idx]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))

    vmax = max(imp[valid].max(), 1e-6) if valid.any() else 1.0
    cmo = plt.get_cmap(cmap)

    def _draw_polys(key, sl_key):
        if key not in features or sl_key not in sl:
            return
        polys = _np(features[key])[..., :2]
        start = sl[sl_key][0]
        for e in range(polys.shape[0]):
            node = start + e
            if node >= len(imp) or not valid[node]:
                continue
            m = np.abs(polys[e]).sum(-1) > 1e-6
            if not m.any():
                continue
            w = float(imp[node] / vmax)
            ax.plot(polys[e][m, 0], polys[e][m, 1], '-', color=cmo(w),
                    linewidth=1.0 + 4.5 * w, alpha=0.25 + 0.7 * w, zorder=2 + int(8 * w))

    _draw_polys('map_lanes', 'lane')
    _draw_polys('route_lanes', 'route')
    _draw_polys('map_crosswalks', 'crosswalk')

    type_marker = {NODE_TYPE_VEHICLE: 'o', NODE_TYPE_PEDESTRIAN: 'P', NODE_TYPE_BICYCLE: 'X'}
    a0, a1 = sl['agent']
    for v in range(a0, a1):
        if not valid[v]:
            continue
        x, y = pose[v, 0], pose[v, 1]
        if ntypes[v] == NODE_TYPE_EGO:
            ax.scatter(x, y, c='deepskyblue', s=280, marker='*', edgecolors='k', linewidths=1.0, zorder=20)
        else:
            mk = type_marker.get(int(ntypes[v]), 'o')
            ax.scatter(x, y, c=[imp[v]], cmap=cmap, vmin=0, vmax=vmax,
                       s=120 + 450 * (imp[v] / vmax), marker=mk,
                       edgecolors='k', linewidths=0.9, zorder=21)

    def _topk(rng, k):
        idxs = [v for v in range(rng[0], rng[1]) if valid[v]]
        return sorted(idxs, key=lambda v: -imp[v])[:k]

    map_range = (sl['lane'][0], sl['route'][1])
    for v in _topk(sl['agent'], top_k_each) + _topk(map_range, top_k_each):
        if ntypes[v] == NODE_TYPE_EGO:
            continue
        ax.annotate(f"{imp[v]:.2f}", (pose[v, 0], pose[v, 1]), fontsize=8, color='navy', zorder=22,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='navy', alpha=0.8, lw=0.5))

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='importance (dark = high)')

    ax.set_aspect('equal')
    ax.set_xlabel('x (m, ego frame)')
    ax.set_ylabel('y (m, ego frame)')
    ax.set_title(title or 'Importance map: what the vehicle attended to when deciding (dark = important)')

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    return ax

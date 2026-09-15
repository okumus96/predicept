import math

import torch

CH_SAME_LANE_AHEAD = 0
CH_SAME_LANE_BEHIND = 1
CH_ADJACENT_LEFT = 2
CH_ADJACENT_RIGHT = 3
CH_COLLISION_COURSE = 4
CH_SHARES_INTERSECTION = 5
CH_NEAR = 6
CH_FOLLOWS = 7
CH_MERGES = 8
CH_OVERTAKES = 9
CH_VRU = 10
NUM_CHANNELS = 11

CHANNEL_NAMES = [
    "same_lane_ahead", "same_lane_behind", "adjacent_left", "adjacent_right",
    "onObservedCollisionCourseWith", "sharesIntersectionWith", "near",
    "follows", "merges", "overtakes", "vulnerable_road_user_near_ego_path",
]

EV_DS = 0
EV_DLAT = 1
EV_DFS = 2
EV_CLOSING = 3
EV_TTC = 4
EV_T_ENTRY = 5
EV_DTHETA_FLOW = 6
EV_VLAT = 7
EV_DS_ENTRY = 8
NUM_EVIDENCE = 9

LANE_W = 3.5
AHEAD_MAX_M = 80.0
BEHIND_MAX_M = 25.0
ADJ_MAX_M = 35.0
SAME_FLOW_RAD = 0.45
MAP_DIR_RAD = 0.60
NEAR_M = 5.0
CPA_CLEARANCE_M = 1.0
HORIZON_S = 8.0
DT = 0.1
CROSSING_LOOKAHEAD_M = 60.0
CROSSING_MIN_ANGLE = math.radians(30.0)  # *
MERGE_VLAT_MIN = 0.2
OT_SIDE_MIN_LAT = 1.25
OT_REL_SPEED_MIN = 0.3
OT_CLEAR_M = 1.0
OT_WINDOW_DS_M = 10.0
OT_PAST_ADVANCE_M = 1.0
FOL_HEADWAY_S = 5.0
FOL_MAX_GAP_M = 80.0
FOL_QUEUE_VS = 2.0
FOL_QUEUE_VO = 4.0
FOL_QUEUE_GAP_M = 12.0
FOL_MARGIN_M = 0.50
FOL_PERSIST_FRAMES = 11
EGO_HALF_LEN_M = 2.31
VRU_CRIT_M = 12.0


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def select_ego_corridor(ref_path):
    xy = ref_path[..., :2].float()                                  # [B,R,P,2]
    yaw = ref_path[..., 2].float()
    valid = xy.abs().sum(-1) > 1e-6                                 # [B,R,P]
    d = xy.norm(dim=-1).masked_fill(~valid, 1e9)
    idx = d.argmin(dim=-1)
    g = lambda t: torch.gather(t, 2, idx.unsqueeze(-1)).squeeze(-1)
    y0, px, py = g(yaw), g(xy[..., 0]), g(xy[..., 1])
    dlat = (torch.sin(y0) * px - torch.cos(y0) * py).abs()
    hd = torch.atan2(torch.sin(y0), torch.cos(y0)).abs()
    cand_ok = valid.any(-1)
    inlane = cand_ok & (dlat <= LANE_W / 2) & (hd <= MAP_DIR_RAD)
    seg = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1)               # [B,R,P-1]
    cum = torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(-1)], dim=-1)  # [B,R,P]
    near20 = valid & (cum <= 20.0)
    hd_all = torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs()      # [B,R,P]
    hd20 = ((hd_all * near20).sum(-1) / near20.sum(-1).clamp(min=1))
    smax = torch.where(valid, cum, torch.zeros_like(cum)).amax(-1)
    score = dlat + 0.3 * hd20 + 1e-3 * (200.0 - smax.clamp(max=200.0))
    pick_fb = score.masked_fill(~cand_ok, 1e9).argmin(dim=-1)
    pick_in = score.masked_fill(~inlane, 1e9).argmin(dim=-1)
    return torch.where(inlane.any(-1), pick_in, pick_fb)            # [B]


def _corridor_arrays(ref_path):
    sel = select_ego_corridor(ref_path)                             # [B]
    r0 = ref_path[torch.arange(ref_path.shape[0], device=ref_path.device), sel]
    xy = r0[..., :2].float()
    yaw = r0[..., 2].float()
    valid = xy.abs().sum(-1) > 1e-6
    seg = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(dim=1)], dim=1)  # [B,P]
    return xy, yaw, cum, valid


def _project(points, cxy, cyaw, ccum, cvalid):
    d = (points[:, :, None, :] - cxy[:, None, :, :]).norm(dim=-1)          # [B,K,P]
    d = d.masked_fill(~cvalid[:, None, :], 1e9)
    idx = d.argmin(dim=-1)                                                 # [B,K]
    s = torch.gather(ccum, 1, idx)
    yawk = torch.gather(cyaw, 1, idx)
    cx = torch.gather(cxy[..., 0], 1, idx)
    cy = torch.gather(cxy[..., 1], 1, idx)
    dx = points[..., 0] - cx
    dy = points[..., 1] - cy
    d_lat = -torch.sin(yawk) * dx + torch.cos(yawk) * dy
    on_start = idx == 0
    return s, d_lat, yawk, on_start


def _arc_walk(cxy, ccum, cvalid, s_query):
    B, P = ccum.shape
    smax = torch.where(cvalid, ccum, torch.zeros_like(ccum)).max(dim=1, keepdim=True).values
    sq = s_query.clamp(min=0.0)
    sq = torch.minimum(sq, smax.expand_as(sq))
    idx = torch.searchsorted(ccum.contiguous(), sq.contiguous(), right=True).clamp(1, P - 1)
    s1 = torch.gather(ccum, 1, idx - 1)
    s2 = torch.gather(ccum, 1, idx)
    w = ((sq - s1) / (s2 - s1).clamp(min=1e-6)).unsqueeze(-1)
    p1 = torch.gather(cxy, 1, (idx - 1).unsqueeze(-1).expand(-1, -1, 2))
    p2 = torch.gather(cxy, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    return p1 + w * (p2 - p1)


@torch.no_grad()
def compute_channels(neighbor_agents_past, ego_agent_past, neighbor_futures, ref_path,
                     neighbor_valid=None):
    B, N, _, _ = neighbor_agents_past.shape
    dev = neighbor_agents_past.dtype
    cur = neighbor_agents_past[:, :, -1]                       # [B,N,11]
    pos = cur[..., 0:2]
    theta = cur[..., 2]
    vel = cur[..., 3:5]
    L = cur[..., 6].clamp(min=0.5)
    W = cur[..., 7].clamp(min=0.3)
    if neighbor_valid is None:
        neighbor_valid = cur[..., :2].abs().sum(-1) > 1e-6

    ego_cur = ego_agent_past[:, -1]
    ego_v = ego_cur[..., 3:5].norm(dim=-1)                      # [B]
    r_ego = 0.5 * math.hypot(4.6, 2.0)
    r_j = 0.5 * torch.sqrt(L ** 2 + W ** 2)

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
    has_corr = cvalid.any(dim=1)                                # [B]

    s_j, d_lat, tan_yaw, on_start = _project(pos, cxy, cyaw, ccum, cvalid)
    behind_mode = on_start & (pos[..., 0] < 0)
    ds = torch.where(behind_mode, pos[..., 0], s_j)
    d_lat_eff = torch.where(behind_mode, pos[..., 1], d_lat)
    tan_eff = torch.where(behind_mode, torch.zeros_like(tan_yaw), tan_yaw)

    dtheta_flow = _wrap(theta - tan_eff).abs()
    same_flow = dtheta_flow <= SAME_FLOW_RAD
    d_center = pos.norm(dim=-1)
    d_fs = d_center - r_ego - r_j
    ego_vv = ego_cur[..., 3:5].unsqueeze(1)
    closing = -((pos * (vel - ego_vv)).sum(-1) / d_center.clamp(min=1e-3))
    v_lat_corr = -torch.sin(tan_eff) * vel[..., 0] + torch.cos(tan_eff) * vel[..., 1]
    v_lat_toward = -torch.sign(d_lat_eff) * v_lat_corr

    T = neighbor_futures.shape[2]
    tgrid = torch.arange(1, T + 1, device=neighbor_agents_past.device).float() * DT   # [T]
    ego_sweep = _arc_walk(cxy, ccum, cvalid, ego_v[:, None] * tgrid[None, :])          # [B,T,2]
    fut = neighbor_futures[..., :2].float()                                            # [B,N,T,2]
    fut_valid = fut.abs().sum(-1) > 1e-6                                               # [B,N,T]
    d_align = (fut - ego_sweep[:, None]).norm(dim=-1).masked_fill(~fut_valid, 1e9)     # [B,N,T]
    fs, fdlat, _, _ = _project(fut.reshape(B, N * T, 2), cxy, cyaw, ccum, cvalid)
    fs = fs.view(B, N, T)
    fdlat = fdlat.view(B, N, T)

    tip = cur[..., 8:11].argmax(-1)
    veh_like = tip != 1
    motor = tip == 0

    active = torch.zeros(B, N, NUM_CHANNELS, dtype=torch.bool, device=pos.device)
    inlane = d_lat_eff.abs() <= 0.5 * LANE_W
    adjL = (d_lat_eff > 0.5 * LANE_W) & (d_lat_eff <= 1.5 * LANE_W)
    adjR = (d_lat_eff < -0.5 * LANE_W) & (d_lat_eff >= -1.5 * LANE_W)

    active[..., CH_SAME_LANE_AHEAD] = inlane & same_flow & (ds > 0) & (ds <= AHEAD_MAX_M) & veh_like
    active[..., CH_SAME_LANE_BEHIND] = inlane & same_flow & (ds < 0) & (ds >= -BEHIND_MAX_M) & veh_like
    adj_lon_ok = ds >= -BEHIND_MAX_M
    active[..., CH_ADJACENT_LEFT] = adjL & same_flow & (d_center <= ADJ_MAX_M) & adj_lon_ok & veh_like
    active[..., CH_ADJACENT_RIGHT] = adjR & same_flow & (d_center <= ADJ_MAX_M) & adj_lon_ok & veh_like

    r_w = 1.0 + 0.5 * W                                                                # [B,N]
    clear = d_align - r_w[..., None]
    hit = clear <= CPA_CLEARANCE_M                                                     # [B,N,T]
    ttc = torch.where(hit.any(-1),
                      tgrid[None, None, :].expand_as(hit).masked_fill(~hit, HORIZON_S).min(-1).values,
                      torch.full_like(d_center, HORIZON_S))
    cc_directional = (ds > 0) | (dtheta_flow > CROSSING_MIN_ANGLE)
    active[..., CH_COLLISION_COURSE] = hit.any(-1) & (closing > 0) & cc_directional

    fut_head = torch.cat([fut[:, :, 1:] - fut[:, :, :-1], fut[:, :, -1:] - fut[:, :, -2:-1]], dim=2)
    fut_ang = torch.atan2(fut_head[..., 1], fut_head[..., 0])
    on_corr = (fdlat.abs() <= 0.5 * LANE_W) & (fs > 0) & (fs <= CROSSING_LOOKAHEAD_M)
    cross_ang = _wrap(fut_ang - tan_eff[..., None]).abs()
    crossing_pt = on_corr & (cross_ang > CROSSING_MIN_ANGLE) & (cross_ang < math.pi - CROSSING_MIN_ANGLE)
    active[..., CH_SHARES_INTERSECTION] = crossing_pt.any(-1)

    t_entry_hit = (fdlat.abs() <= 0.5 * LANE_W) & (fs > 0) & fut_valid
    t_entry = torch.where(t_entry_hit.any(-1),
                          tgrid[None, None, :].expand_as(t_entry_hit).masked_fill(~t_entry_hit, HORIZON_S).min(-1).values,
                          torch.full_like(d_center, HORIZON_S))
    idx_entry = t_entry_hit.float().argmax(-1)
    fs_entry = torch.gather(fs, 2, idx_entry.unsqueeze(-1)).squeeze(-1)
    ds_entry = torch.where(t_entry_hit.any(-1), fs_entry - ego_v[:, None] * t_entry,
                           torch.zeros_like(d_center))
    active[..., CH_MERGES] = (adjL | adjR) & (v_lat_toward >= MERGE_VLAT_MIN) & t_entry_hit.any(-1) & veh_like

    past_rel = neighbor_agents_past[..., 0] - ego_agent_past[..., 0].unsqueeze(1)

    g = ds - 0.5 * L - EGO_HALF_LEN_M
    v_o = vel.norm(dim=-1)
    ego_vN = ego_v[:, None]
    moving_f = (ego_vN > 0.30) & (g > 0) & (g <= FOL_MAX_GAP_M) & (g <= FOL_HEADWAY_S * ego_vN)
    queue_f = (ego_vN <= FOL_QUEUE_VS) & (v_o <= FOL_QUEUE_VO) & (g > 0) & (g <= FOL_QUEUE_GAP_M)
    fol_cand = inlane & same_flow & (ds > 0) & motor & (moving_f | queue_f) & neighbor_valid
    ds_c = ds.masked_fill(~fol_cand, 1e9)
    best_v, best_i = ds_c.min(dim=-1, keepdim=True)
    second_v = ds_c.scatter(1, best_i, torch.full_like(best_v, 1e9)).min(dim=-1, keepdim=True).values
    unique_leader = (second_v - best_v) > FOL_MARGIN_M
    ahead_1s = (past_rel[..., -FOL_PERSIST_FRAMES:] > 0).all(-1)
    active[..., CH_FOLLOWS] = fol_cand & (ds_c == best_v) & unique_leader & ahead_1s

    past_adv = past_rel[..., -1] - past_rel[..., 0]
    ds_fut = fs - (ego_v[:, None, None] * tgrid[None, None, :])
    v_long_rel = ((vel - ego_vv) * torch.stack([torch.cos(tan_eff), torch.sin(tan_eff)], -1)).sum(-1)
    was_not_ahead = past_rel[..., 0] <= OT_PAST_ADVANCE_M
    ot_completes = ((ds_fut >= OT_CLEAR_M) & (fdlat.abs() <= 0.5 * LANE_W) & fut_valid).any(-1)
    active[..., CH_OVERTAKES] = ((past_adv >= OT_PAST_ADVANCE_M)
                                  & was_not_ahead
                                  & (ds.abs() <= OT_WINDOW_DS_M)
                                  & (d_lat_eff.abs() >= OT_SIDE_MIN_LAT)
                                  & (v_long_rel >= OT_REL_SPEED_MIN)
                                  & ot_completes
                                  & motor)

    active[..., CH_VRU] = (tip != 0) & (d_center <= VRU_CRIT_M) & (closing > 0)

    none_yet = ~active.any(-1)
    active[..., CH_NEAR] = none_yet & (d_fs <= NEAR_M)

    no_corr = ~has_corr
    if no_corr.any():
        keep = torch.zeros_like(active)
        keep[..., CH_NEAR] = d_fs <= NEAR_M
        keep[..., CH_VRU] = (tip != 0) & (d_center <= VRU_CRIT_M) & (closing > 0)
        active[no_corr] = keep[no_corr]

    active = active & neighbor_valid.unsqueeze(-1)

    evidence = torch.stack([
        ds, d_lat_eff, d_fs, closing, ttc, t_entry, dtheta_flow, v_lat_toward, ds_entry,
    ], dim=-1).to(dev)
    evidence = evidence * neighbor_valid.unsqueeze(-1)

    return active, evidence


MCH_IN_LANE = 0
MCH_ADJ_LEFT = 1
MCH_ADJ_RIGHT = 2
MCH_SUCCESSOR = 3
MCH_SHARES_INT = 4
MCH_ROUTE = 5
MCH_TRAFFIC = 6
MCH_NEAR = 7
NUM_MAP_CHANNELS = 8
MAP_CHANNEL_NAMES = ["inLane", "adjacent_left", "adjacent_right", "successor",
                     "inIntersection", "ego_route_corridor", "traffic_control", "near"]
NUM_MAP_EVIDENCE = 8


@torch.no_grad()
def compute_map_channels(map_lanes, map_crosswalks, route_lanes, ref_path):
    B = map_lanes.shape[0]
    dev = map_lanes.device

    def _elems(t):
        xy = t[..., :2].float()
        hd = t[..., 2].float()
        valid = xy.abs().sum(-1) > 1e-6                     # [B,E,P]
        return xy, hd, valid

    lx, lh, lv = _elems(map_lanes)
    cx, ch_, cv = _elems(map_crosswalks)
    rx, rh, rv = _elems(route_lanes)
    tl = map_lanes[..., 3:7].float()                        # [B,L,P,4]

    Pmax = max(lx.shape[2], cx.shape[2], rx.shape[2])

    def _pad(x, val=0.0):
        if x.shape[2] == Pmax:
            return x
        pad_shape = list(x.shape)
        pad_shape[2] = Pmax - x.shape[2]
        return torch.cat([x, torch.full(pad_shape, val, dtype=x.dtype, device=x.device)], dim=2)

    exy = torch.cat([_pad(lx), _pad(cx), _pad(rx)], dim=1)
    ehd = torch.cat([_pad(lh), _pad(ch_), _pad(rh)], dim=1)
    ev_ = torch.cat([_pad(lv.float()), _pad(cv.float()), _pad(rv.float())], dim=1) > 0.5
    S = exy.shape[1]
    L = lx.shape[1]
    C = cx.shape[1]
    elem_valid = ev_.any(-1)                                        # [B,S]
    is_route_tok = torch.zeros(B, S, dtype=torch.bool, device=dev)
    is_route_tok[:, L + C:] = True
    is_cross_tok = torch.zeros(B, S, dtype=torch.bool, device=dev)
    is_cross_tok[:, L:L + C] = True

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)

    flat = exy.reshape(B, S * Pmax, 2)
    fs, fdlat, fyaw, _ = _project(flat, cxy, cyaw, ccum, cvalid)
    fs = fs.view(B, S, Pmax)
    fdlat = fdlat.view(B, S, Pmax)
    big = torch.tensor(1e9, device=dev)
    fdlat_m = torch.where(ev_, fdlat, big)

    d_ego = torch.where(ev_, exy.norm(dim=-1), big)
    min_d_ego, min_idx = d_ego.min(dim=-1)                          # [B,S]
    hd_near = torch.gather(ehd, 2, min_idx.unsqueeze(-1)).squeeze(-1)

    in_band_ego = min_d_ego <= 0.5 * LANE_W
    aligned_ego = torch.cos(hd_near) > math.cos(MAP_DIR_RAD)
    inlane = in_band_ego & aligned_ego & ~is_cross_tok

    on_corr_pt = (fdlat_m.abs() <= 0.5 * LANE_W) & (fs > 1.0)
    frac_on_corr = (on_corr_pt & ev_).float().sum(-1) / ev_.float().sum(-1).clamp(min=1.0)
    successor = (frac_on_corr > 0.3) & ~inlane & ~is_cross_tok

    med_dlat = torch.where(ev_, fdlat, torch.zeros_like(fdlat)).sum(-1) / ev_.float().sum(-1).clamp(min=1.0)
    tan_near = torch.gather(fyaw.view(B, S, Pmax), 2, min_idx.unsqueeze(-1)).squeeze(-1)
    par = torch.cos(hd_near - tan_near) > math.cos(MAP_DIR_RAD)
    reaches_ego = torch.where(ev_, exy[..., 0], torch.tensor(-1e9, device=dev)).amax(-1) > 0.0
    adjL = (med_dlat > 0.5 * LANE_W) & (med_dlat <= 1.5 * LANE_W) & par & reaches_ego & ~is_cross_tok
    adjR = (med_dlat < -0.5 * LANE_W) & (med_dlat >= -1.5 * LANE_W) & par & reaches_ego & ~is_cross_tok

    has_tl = torch.zeros(B, S, dtype=torch.bool, device=dev)
    has_tl[:, :L] = tl[..., :3].abs().sum(-1).amax(-1) > 1e-6
    tl_onehot = torch.zeros(B, S, 4, device=dev)
    tl_onehot[:, :L] = tl.amax(dim=2)

    route_fire = is_route_tok & reaches_ego

    active = torch.zeros(B, S, NUM_MAP_CHANNELS, dtype=torch.bool, device=dev)
    active[..., MCH_IN_LANE] = inlane
    active[..., MCH_ADJ_LEFT] = adjL
    active[..., MCH_ADJ_RIGHT] = adjR
    active[..., MCH_SUCCESSOR] = successor
    active[..., MCH_ROUTE] = route_fire
    active[..., MCH_TRAFFIC] = has_tl & (inlane | successor | is_route_tok)
    none_yet = ~active.any(-1)
    active[..., MCH_NEAR] = none_yet & (min_d_ego <= 20.0)
    active = active & elem_valid.unsqueeze(-1)

    s_near = torch.gather(fs, 2, min_idx.unsqueeze(-1)).squeeze(-1)
    evidence = torch.cat([
        min_d_ego.unsqueeze(-1), med_dlat.unsqueeze(-1), s_near.unsqueeze(-1),
        _wrap(hd_near - tan_near).abs().unsqueeze(-1), tl_onehot,
    ], dim=-1)
    evidence = evidence * elem_valid.unsqueeze(-1)
    return active, evidence

import math
import torch

from .channels import (_wrap, _project, _corridor_arrays, _arc_walk,
                       LANE_W, SAME_FLOW_RAD, MAP_DIR_RAD, DT,
                       compute_channels as _compute_channels_v2, CH_COLLISION_COURSE)

A_SAME_LANE_AHEAD = 0
A_SAME_LANE_BEHIND = 1
A_LEFT_ADJACENT = 2
A_RIGHT_ADJACENT = 3
A_VRU_NEAR_PATH = 4
A_CONFLICTS_WITH_PATH = 5
A_PREDICTED_CLOSE_APPROACH = A_CONFLICTS_WITH_PATH
NUM_A = 6
ADJ_ON_LANE_M = 2.0
A_SHARES_INTERSECTION = None
A_NAMES = ["same_lane_ahead", "same_lane_behind", "leftAdjacentAgent", "rightAdjacentAgent",
           "VRU_near_ego_path", "conflictsWithPath"]

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

M_IN_LANE = 0
M_LEFT_ADJACENT = 1
M_RIGHT_ADJACENT = 2
M_SUCCESSOR = 3
M_CROSSWALK_ON_PATH = 4
M_CROSSES_EGO_PATH = M_CROSSWALK_ON_PATH
M_STOP_LINE = 5
M_RED_LIGHT = 6
M_GREEN_LIGHT = 7
M_TRAFFIC_CONTROL = M_STOP_LINE
NUM_M = 8
M_NAMES = ["inLane", "leftAdjacentLane", "rightAdjacentLane", "successor",
           "crosswalkOnPath", "stopLine", "redLight", "greenLight"]

AHEAD_MAX_M = 80.0
BEHIND_MAX_M = 25.0
ADJ_MAX_M = 35.0
VRU_MAX_M = 35.0
SHARED_INT_MAX_M = 55.0
LON_TAU_S = 5.0
LON_MIN_M = 25.0
LON_BEHIND_M = 10.0
EGO_L, EGO_W = 4.62, 2.10
STATIC_SPEED = 0.5
STATIC_MAX_M = 60.0
ROUTE_LAT_TOL = 1.75
CROSSWALK_TOL = 2.0
CW_LOOKAHEAD_M = 50.0
TL_STOP_TOL = 3.0
ENTER_LAT_M = 2.5
STOP_START_TOL = 5.0
CROSS_LAT_M = 8.0
CROSS_ANG_LO = 0.5236
CROSS_ANG_HI = 2.6180
JUNC_LOOKAHEAD_M = 40.0
CORR_STRIDE = 10
CROSS_MIN_V = 0.5


def _poly_valid(poly):
    pv = poly[..., :2].abs().sum(-1) > 1e-6
    return pv, pv.any(-1)


def _point_in_polys(pts, poly, poly_pt_valid, poly_valid):
    B, N, _ = pts.shape
    K, P = poly.shape[1], poly.shape[2]
    first = poly[..., :1, :2]
    pxy = torch.where(poly_pt_valid.unsqueeze(-1), poly[..., :2], first.expand(-1, -1, P, -1))
    px, py = pxy[..., 0], pxy[..., 1]                         # [B,K,P]
    nxt = torch.roll(px, -1, dims=2), torch.roll(py, -1, dims=2)
    x = pts[:, :, None, None, 0]                              # [B,N,1,1]
    y = pts[:, :, None, None, 1]
    x1, y1 = px[:, None], py[:, None]                         # [B,1,K,P]
    x2, y2 = nxt[0][:, None], nxt[1][:, None]
    cond = ((y1 > y) != (y2 > y))
    denom = (y2 - y1)
    denom = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    xint = x1 + (y - y1) * (x2 - x1) / denom
    crossings = (cond & (x < xint)).sum(-1)                   # [B,N,K]
    return (crossings % 2 == 1) & poly_valid[:, None]


def _footprint_pts(pos, theta, length, width):
    c, sn = torch.cos(theta), torch.sin(theta)
    hl, hw = 0.5 * length, 0.5 * width
    ox = torch.stack([torch.zeros_like(hl), hl, hl, -hl, -hl], dim=-1)      # [B,N,5]
    oy = torch.stack([torch.zeros_like(hw), hw, -hw, hw, -hw], dim=-1)
    gx = pos[..., 0:1] + ox * c.unsqueeze(-1) - oy * sn.unsqueeze(-1)
    gy = pos[..., 1:2] + ox * sn.unsqueeze(-1) + oy * c.unsqueeze(-1)
    return torch.stack([gx, gy], dim=-1)                                    # [B,N,5,2]


def _entity_in_polys(fp, poly, ppv, pv):
    B, N, P5, _ = fp.shape
    hit = _point_in_polys(fp.reshape(B, N * P5, 2), poly, ppv, pv)          # [B,N*5,K]
    return hit.view(B, N, P5, -1).any(2)


def _pv_polyline(xy):
    nz = xy.abs().sum(-1) > 1e-6
    lead = torch.cummax(nz.int(), dim=-1)[0].bool()
    trail = torch.flip(torch.cummax(torch.flip(nz.int(), dims=[-1]), dim=-1)[0],
                       dims=[-1]).bool()
    return lead & trail


def _seg_dist_to_origin(xy, pv):
    p1, p2 = xy[:, :, :-1], xy[:, :, 1:]                       # [B,S,P-1,2]
    seg_ok = pv[:, :, :-1] & pv[:, :, 1:]
    dv = p2 - p1
    L2 = (dv * dv).sum(-1).clamp(min=1e-9)
    t = (-(p1 * dv).sum(-1) / L2).clamp(0.0, 1.0)
    close = p1 + t.unsqueeze(-1) * dv
    dseg = close.norm(dim=-1).masked_fill(~seg_ok, 1e9)        # [B,S,P-1]
    dmin, imin = dseg.min(-1)
    return dmin, imin


def junction_crossing_conflict(neighbor_agents_past, ego_agent_past, ref_path, intersections,
                              neighbor_valid=None):
    cur = neighbor_agents_past[:, :, -1]
    pos, theta = cur[..., 0:2], cur[..., 2]
    if neighbor_valid is None:
        neighbor_valid = cur[..., :2].abs().sum(-1) > 1e-6
    veh_like = cur[..., 8:11].argmax(-1) != 1
    moving = cur[..., 3:5].norm(dim=-1) >= CROSS_MIN_V
    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
    s_j, d_lat, tan_yaw, on_start = _project(pos, cxy, cyaw, ccum, cvalid)
    behind = on_start & (pos[..., 0] < 0)
    ds = torch.where(behind, pos[..., 0], s_j)
    d_lat_eff = torch.where(behind, pos[..., 1], d_lat)
    tan_eff = torch.where(behind, torch.zeros_like(tan_yaw), tan_yaw)
    dth_c = _wrap(theta - tan_eff).abs()
    v_ego = ego_agent_past[:, -1, 3:5].norm(dim=-1)
    lon_ahead = (v_ego * LON_TAU_S).clamp(min=LON_MIN_M, max=AHEAD_MAX_M).unsqueeze(-1)
    ipv, iv = _poly_valid(intersections)
    cs_ = ccum[:, ::CORR_STRIDE]
    corr_in_junc = _point_in_polys(cxy[:, ::CORR_STRIDE], intersections, ipv, iv).any(-1)
    junction_ahead = (corr_in_junc & cvalid[:, ::CORR_STRIDE]
                      & (cs_ >= 0.0) & (cs_ <= JUNC_LOOKAHEAD_M)).any(-1)          # [B]
    return (veh_like & moving & (dth_c >= CROSS_ANG_LO) & (dth_c <= CROSS_ANG_HI)
            & (ds > 0) & (ds <= lon_ahead) & (d_lat_eff.abs() <= CROSS_LAT_M)
            & junction_ahead.unsqueeze(-1) & neighbor_valid)


def compute_agent_channels(neighbor_agents_past, ego_agent_past, ref_path,
                           route_lanes, crosswalks, intersections, stop_polygons,
                           lane_tl, map_lanes, neighbor_valid=None, neighbor_futures=None):
    B, N = neighbor_agents_past.shape[:2]
    dev = neighbor_agents_past.device
    cur = neighbor_agents_past[:, :, -1]
    pos, theta, vel = cur[..., 0:2], cur[..., 2], cur[..., 3:5]
    if neighbor_valid is None:
        neighbor_valid = cur[..., :2].abs().sum(-1) > 1e-6
    tip = cur[..., 8:11].argmax(-1)
    veh_like = tip != 1
    vru = tip != 0
    speed = vel.norm(dim=-1)

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
    s_j, d_lat, tan_yaw, on_start = _project(pos, cxy, cyaw, ccum, cvalid)
    behind = on_start & (pos[..., 0] < 0)
    ds = torch.where(behind, pos[..., 0], s_j)
    d_lat_eff = torch.where(behind, pos[..., 1], d_lat)
    tan_eff = torch.where(behind, torch.zeros_like(tan_yaw), tan_yaw)
    same_flow = _wrap(theta - tan_eff).abs() <= SAME_FLOW_RAD
    d_center = pos.norm(dim=-1)

    inlane = d_lat_eff.abs() <= 0.5 * LANE_W
    adjL = (d_lat_eff > 0.5 * LANE_W) & (d_lat_eff <= 1.5 * LANE_W)
    adjR = (d_lat_eff < -0.5 * LANE_W) & (d_lat_eff >= -1.5 * LANE_W)

    act = torch.zeros(B, N, NUM_A, dtype=torch.bool, device=dev)
    v_ego = ego_agent_past[:, -1, 3:5].norm(dim=-1)                        # [B]
    lon_ahead = (v_ego * LON_TAU_S).clamp(min=LON_MIN_M, max=AHEAD_MAX_M).unsqueeze(-1)
    act[..., A_SAME_LANE_AHEAD] = inlane & same_flow & (ds > 0) & (ds <= lon_ahead) & veh_like
    act[..., A_SAME_LANE_BEHIND] = (inlane & same_flow & (ds < 0)
                                    & (ds >= -LON_BEHIND_M) & veh_like)
    lon_ok = (ds >= -LON_BEHIND_M) & (ds <= lon_ahead)
    act[..., A_LEFT_ADJACENT] = adjL & same_flow & (d_center <= ADJ_MAX_M) & lon_ok & veh_like
    act[..., A_RIGHT_ADJACENT] = adjR & same_flow & (d_center <= ADJ_MAX_M) & lon_ok & veh_like

    ipv, iv = _poly_valid(intersections)
    ego_fp = _footprint_pts(torch.zeros(B, 1, 2, device=dev),
                            torch.zeros(B, 1, device=dev),
                            torch.full((B, 1), EGO_L, device=dev),
                            torch.full((B, 1), EGO_W, device=dev))               # [B,1,5,2]
    ego_in = _entity_in_polys(ego_fp, intersections, ipv, iv)[:, 0]              # [B,I]
    ag_fp = _footprint_pts(pos, theta,
                           cur[..., 6].clamp(min=1.0), cur[..., 7].clamp(min=0.6))
    agent_in = _entity_in_polys(ag_fp, intersections, ipv, iv)                   # [B,N,I]
    kg_shared = (agent_in & ego_in[:, None]).any(-1) & (d_center <= SHARED_INT_MAX_M)

    lxy = map_lanes[..., :2]                                        # [B,L,P,2]
    lpv = _pv_polyline(lxy)
    L_, P_ = lxy.shape[1], lxy.shape[2]
    ls, llat, ltan, _ = _project(lxy.reshape(B, -1, 2), cxy, cyaw, ccum, cvalid)
    ls, llat, ltan = ls.view(B, L_, P_), llat.view(B, L_, P_), ltan.view(B, L_, P_)
    lhd = map_lanes[..., 2]
    sgn = torch.sign(llat)
    pairv = lpv[:, :, :-1] & lpv[:, :, 1:]
    flips = (pairv & (sgn[:, :, :-1] * sgn[:, :, 1:] < 0)
             & (ls[:, :, :-1] > -5.0) & (ls[:, :, :-1] <= AHEAD_MAX_M)
             & (_wrap(lhd - ltan)[:, :, :-1].abs() > SAME_FLOW_RAD))
    lane_crosses = flips.any(-1)                                    # [B,L]

    d_al = torch.cdist(pos, lxy.reshape(B, -1, 2)).masked_fill(
        ~lpv.reshape(B, 1, -1), 1e9).view(B, N, L_, P_)             # [B,N,L,P]
    dmin, imin = d_al.min(-1)                                       # [B,N,L]
    hd_at = torch.gather(lhd.unsqueeze(1).expand(-1, N, -1, -1), 3,
                         imin.unsqueeze(-1)).squeeze(-1)            # [B,N,L]
    align = _wrap(theta.unsqueeze(-1) - hd_at).abs() <= math.radians(60.0)
    cost = dmin + (~align).float() * 1e6 + (dmin > 3.0).float() * 1e6
    best_cost, best_lane = cost.min(-1)                             # [B,N]
    has_lane = best_cost < 1e6
    _A = lxy[:, :, :-1]; _Bp = lxy[:, :, 1:]                                # [B,L,P-1,2]
    _ok = lpv[:, :, :-1] & lpv[:, :, 1:]
    _d = _Bp - _A; _L2 = (_d * _d).sum(-1).clamp(min=1e-9)                  # [B,L,P-1]
    _rel = pos[:, :, None, None, :] - _A[:, None]                           # [B,N,L,P-1,2]
    _t = ((_rel * _d[:, None]).sum(-1) / _L2[:, None]).clamp(0.0, 1.0)
    _C = _A[:, None] + _t.unsqueeze(-1) * _d[:, None]
    _ds = (pos[:, :, None, None, :] - _C).norm(dim=-1)                      # [B,N,L,P-1]
    _al = _wrap(theta[:, :, None, None] - lhd[:, None, :, :-1]).abs() <= math.radians(60.0)
    _ds = _ds.masked_fill(~(_ok[:, None] & _al), 1e9)
    on_lane = _ds.amin(dim=(2, 3)) <= ADJ_ON_LANE_M                          # [B,N]
    v_ego_ = ego_agent_past[:, -1, 3:5].norm(dim=-1)
    map_act_ = compute_map_channels_v3(map_lanes, crosswalks, route_lanes, ref_path,
                                       lane_tl, intersections, stop_polygons, ego_v=v_ego_,
                                       ego_past=ego_agent_past)
    _bl = best_lane.clamp(min=0)
    lane_is_L = torch.gather(map_act_[:, :L_, M_LEFT_ADJACENT], 1, _bl) & has_lane & on_lane
    lane_is_R = torch.gather(map_act_[:, :L_, M_RIGHT_ADJACENT], 1, _bl) & has_lane & on_lane
    act[..., A_LEFT_ADJACENT] &= lane_is_L
    act[..., A_RIGHT_ADJACENT] &= lane_is_R
    my_lane_crosses = torch.gather(lane_crosses.unsqueeze(1).expand(-1, N, -1),
                                   2, best_lane.unsqueeze(-1)).squeeze(-1)

    dth_c = _wrap(theta - tan_eff).abs()
    heading_crossing = (dth_c > SAME_FLOW_RAD) & (dth_c < math.pi - SAME_FLOW_RAD)
    conflict = torch.where(has_lane, my_lane_crosses, heading_crossing)

    _ = kg_shared & conflict

    act[..., A_VRU_NEAR_PATH] = (vru & (d_center <= VRU_MAX_M)
                                 & (d_lat_eff.abs() <= 1.5 * LANE_W) & (ds >= -5.0))


    act[..., A_CONFLICTS_WITH_PATH] = junction_crossing_conflict(
        neighbor_agents_past, ego_agent_past, ref_path, intersections, neighbor_valid)
    if neighbor_futures is not None:
        v2_act, _ = _compute_channels_v2(neighbor_agents_past, ego_agent_past, neighbor_futures, ref_path)
        act[..., A_CONFLICTS_WITH_PATH] |= v2_act[..., CH_COLLISION_COURSE]
    return act & neighbor_valid.unsqueeze(-1)


def compute_map_channels_v3(map_lanes, map_crosswalks, route_lanes, ref_path,
                            lane_tl, intersections, stop_polygons, ego_v=None, return_aux=False,
                            ego_past=None):
    B, L, P, _ = map_lanes.shape
    dev = map_lanes.device
    C, R = map_crosswalks.shape[1], route_lanes.shape[1]
    Pm = max(P, map_crosswalks.shape[2], route_lanes.shape[2])

    def pad(t):
        if t.shape[2] == Pm:
            return t
        z = torch.zeros(t.shape[0], t.shape[1], Pm - t.shape[2], t.shape[3],
                        dtype=t.dtype, device=t.device)
        return torch.cat([t, z], dim=2)

    exy = torch.cat([pad(map_lanes[..., :3]), pad(map_crosswalks[..., :3]),
                     pad(route_lanes[..., :3])], dim=1)                      # [B,S,Pm,3]
    S = exy.shape[1]
    ehd = exy[..., 2]
    xy = exy[..., :2]
    pv = _pv_polyline(xy)
    elem_valid = pv.any(-1)
    is_cw = torch.zeros(B, S, dtype=torch.bool, device=dev); is_cw[:, L:L + C] = True
    is_rt = torch.zeros(B, S, dtype=torch.bool, device=dev); is_rt[:, L + C:] = True

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
    min_d, min_i = _seg_dist_to_origin(xy, pv)
    hd_near = torch.gather(ehd, 2, min_i.unsqueeze(-1)).squeeze(-1)
    fs, fdlat, ftan, _ = _project(xy.reshape(B, S * Pm, 2), cxy, cyaw, ccum, cvalid)
    fs, fdlat = fs.view(B, S, Pm), fdlat.view(B, S, Pm)
    tan_near = torch.gather(ftan.view(B, S, Pm), 2, min_i.unsqueeze(-1)).squeeze(-1)

    act = torch.zeros(B, S, NUM_M, dtype=torch.bool, device=dev)
    aligned = torch.cos(hd_near) > math.cos(MAP_DIR_RAD)
    _base_in = (min_d <= 0.5 * LANE_W) & aligned & ~is_cw
    _i0 = pv.float().argmax(-1, keepdim=True)
    _iL = (Pm - 1 - pv.flip(-1).float().argmax(-1)).unsqueeze(-1)
    _fs0 = torch.gather(fs, 2, _i0).squeeze(-1); _fl0 = torch.gather(fdlat, 2, _i0).squeeze(-1)
    _flL = torch.gather(fdlat, 2, _iL).squeeze(-1)
    _p0 = torch.gather(xy, 2, _i0.unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
    _d0 = _p0.norm(dim=-1)
    _seg = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1) * (pv[:, :, :-1] & pv[:, :, 1:]).float()   # [B,S,P-1]
    _cl = torch.cat([torch.zeros(B, S, 1, device=dev), _seg.cumsum(-1)], dim=-1)                   # [B,S,P]
    _s_at = torch.gather(_cl, 2, min_i.unsqueeze(-1)).squeeze(-1)
    _tot = _cl[..., -1]
    interior = (_s_at >= 2.0) & ((_tot - _s_at) >= 2.0)
    starts_here = _d0 <= 5.0
    sel = _base_in.clone()
    multi = sel.sum(-1, keepdim=True) >= 2
    _in_int = sel & interior
    sel = torch.where(multi & (_in_int.sum(-1, keepdim=True) >= 1), _in_int, sel)
    multi2 = (sel.sum(-1, keepdim=True) >= 2) & (_in_int.sum(-1, keepdim=True) >= 2)
    if ego_past is not None and bool(multi2.any()):
        _pp = ego_past[..., :2]
        _A = xy[:, :, :-1]; _Bp = xy[:, :, 1:]; _ok = pv[:, :, :-1] & pv[:, :, 1:]
        _dv = _Bp - _A; _L2 = (_dv * _dv).sum(-1).clamp(min=1e-9)
        _rel = _pp[:, None, None, :, :] - _A[:, :, :, None, :]
        _t = ((_rel * _dv[:, :, :, None, :]).sum(-1) / _L2[..., None]).clamp(0, 1)
        _C = _A[:, :, :, None, :] + _t.unsqueeze(-1) * _dv[:, :, :, None, :]
        _dd = (_pp[:, None, None] - _C).norm(dim=-1).masked_fill(~_ok[..., None], 1e9)
        _past_d = _dd.amin(2).mean(-1)                                                # [B,S]
        _keep = torch.zeros_like(sel); _keep.scatter_(1, _past_d.masked_fill(~sel, 1e9).argmin(-1, keepdim=True), True)
        sel = torch.where(multi2, sel & _keep, sel)
    multi3 = sel.sum(-1, keepdim=True) >= 2
    if bool(multi3.any()):
        _pL = torch.gather(xy, 2, _iL.unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
        _dL = _pL.norm(dim=-1)
        _keep = torch.zeros_like(sel); _keep.scatter_(1, _dL.masked_fill(~sel, 1e9).argmin(-1, keepdim=True), True)
        sel = torch.where(multi3, sel & _keep, sel)
    act[..., M_IN_LANE] = sel

    med = torch.where(pv, fdlat, torch.zeros_like(fdlat)).sum(-1) / pv.float().sum(-1).clamp(min=1)
    par = torch.cos(hd_near - tan_near) > math.cos(MAP_DIR_RAD)
    reach = torch.where(pv, xy[..., 0], torch.full_like(xy[..., 0], -1e9)).amax(-1) > 0
    act[..., M_LEFT_ADJACENT] = (med > 0.5 * LANE_W) & (med <= 1.5 * LANE_W) & par & reach & ~is_cw
    act[..., M_RIGHT_ADJACENT] = (med < -0.5 * LANE_W) & (med >= -1.5 * LANE_W) & par & reach & ~is_cw

    on_corr = (fdlat.abs() <= 0.5 * LANE_W) & (fs > 1.0)
    frac = (on_corr & pv).float().sum(-1) / pv.float().sum(-1).clamp(min=1)
    enters_at_start = (_fl0.abs() <= ENTER_LAT_M) & (_fs0 >= -5.0)
    _fsL = torch.gather(fs, 2, _iL).squeeze(-1)
    _clen = (ccum * cvalid.float()).amax(-1, keepdim=True)                              # [B,1]
    exits_at_end = (_flL.abs() <= ENTER_LAT_M) | (_fsL >= _clen - 2.0)
    act[..., M_SUCCESSOR] = ((frac > 0.3) & enters_at_start & exits_at_end
                             & ~act[..., M_IN_LANE] & ~is_cw)


    sgn = torch.sign(fdlat)
    pairv_m = pv[:, :, :-1] & pv[:, :, 1:]
    ang = _wrap(ehd - ftan.view(B, S, Pm))[:, :, :-1].abs()
    if ego_v is None:
        cross_range = torch.full((B, 1, 1), AHEAD_MAX_M, device=dev)
    else:
        cross_range = (ego_v.reshape(B) * LON_TAU_S).clamp(min=LON_MIN_M, max=AHEAD_MAX_M).view(B, 1, 1)
    mflips = (pairv_m & (sgn[:, :, :-1] * sgn[:, :, 1:] < 0)
              & (fs[:, :, :-1] > 0.0) & (fs[:, :, :-1] <= cross_range)
              & (ang > SAME_FLOW_RAD))
    half_l = 0.5 * LANE_W
    both_sides = ((fdlat >= half_l) & pv).any(-1) & ((fdlat <= -half_l) & pv).any(-1)
    act[..., M_CROSSWALK_ON_PATH] = mflips.any(-1) & elem_valid & both_sides & is_cw

    on_pt = ((fdlat.abs() <= 0.5 * LANE_W) & (fs > 0.0) & (fs <= AHEAD_MAX_M) & pv
             & (_wrap(ehd - ftan.view(B, S, Pm)).abs() <= SAME_FLOW_RAD))
    on_ego_path = (((on_pt.float().sum(-1) / pv.float().sum(-1).clamp(min=1)) >= 0.3)
                   & (enters_at_start | act[..., M_IN_LANE]))                          # [B,S]
    tl_real = torch.zeros(B, S, dtype=torch.bool, device=dev)
    tl_real[:, :L] = (lane_tl[..., :3].abs().sum(-1) > 1e-6).any(-1)
    spv, sv = _poly_valid(stop_polygons)                          # [B,K,Ps], [B,K]
    K_, Ps = stop_polygons.shape[1], stop_polygons.shape[2]
    _ipv, _iv = _poly_valid(intersections)
    _cen0 = (stop_polygons[..., :2] * spv.unsqueeze(-1)).sum(2) / spv.float().sum(2, keepdim=True).clamp(min=1)  # [B,K,2]
    _in_ix = _point_in_polys(_cen0, intersections, _ipv, _iv).any(-1)                # [B,K]
    sv = sv & ~_in_ix
    sp = stop_polygons[..., :2].reshape(B, K_ * Ps, 2)
    sp_v = spv.reshape(B, K_ * Ps)
    _A = xy[:, :, :-1]; _Bp = xy[:, :, 1:]; _ok = pv[:, :, :-1] & pv[:, :, 1:]      # [B,S,Pm-1,2]
    _d = _Bp - _A; _L2 = (_d * _d).sum(-1).clamp(min=1e-9)
    _rel = sp[:, None, None, :, :] - _A[:, :, :, None, :]                             # [B,S,Pm-1,Q,2]
    _t = ((_rel * _d[:, :, :, None, :]).sum(-1) / _L2[..., None]).clamp(0.0, 1.0)
    _C = _A[:, :, :, None, :] + _t.unsqueeze(-1) * _d[:, :, :, None, :]
    _dist = (sp[:, None, None] - _C).norm(dim=-1)                                     # [B,S,Pm-1,Q]
    _sign = torch.sign(_d[:, :, :, None, 0] * _rel[..., 1] - _d[:, :, :, None, 1] * _rel[..., 0])
    _dist = _dist.masked_fill(~(_ok[..., None] & sp_v[:, None, None]), 1e9)
    _dmin, _imin = _dist.min(2)                                                        # [B,S,Q]
    _lat = torch.gather(_sign * _dist.clamp(max=1e8), 2, _imin.unsqueeze(2)).squeeze(2)
    _lat = _lat.view(B, S, K_, Ps); _dmin = _dmin.view(B, S, K_, Ps)
    _valid = spv[:, None]                                                              # [B,1,K,Ps]
    _near = (_dmin <= TL_STOP_TOL) & _valid
    _lat_ok = _lat.masked_fill(~_valid, 0.0)
    straddle = ((_lat_ok.masked_fill(~_valid, 1e9).amin(-1) <= 0.3)
                & (_lat_ok.masked_fill(~_valid, -1e9).amax(-1) >= -0.3))               # [B,S,K]
    _fs_first = torch.gather(fs, 2, pv.float().argmax(-1, keepdim=True)).squeeze(-1)
    _cen = (stop_polygons[..., :2] * spv.unsqueeze(-1)).sum(2) / spv.float().sum(2, keepdim=True).clamp(min=1)  # [B,K,2]
    _fs_c, _, _, _ = _project(_cen, cxy, cyaw, ccum, cvalid)                           # [B,K]
    near_start = (_fs_c[:, None, :] - _fs_first[:, :, None]).abs() <= STOP_START_TOL    # [B,S,K]
    stop_hit = _near.any(-1) & straddle & near_start & sv[:, None]
    touches_stop = stop_hit.any(-1)                                                    # [B,S]
    tl_red = torch.zeros(B, S, dtype=torch.bool, device=dev)
    tl_green = torch.zeros(B, S, dtype=torch.bool, device=dev)
    tl_red[:, :L] = (lane_tl[..., 2].abs() > 1e-6).any(-1)
    tl_green[:, :L] = (lane_tl[..., 0].abs() > 1e-6).any(-1) & ~tl_red[:, :L]
    act[..., M_STOP_LINE] = on_ego_path & touches_stop & elem_valid
    act[..., M_RED_LIGHT] = on_ego_path & tl_red & elem_valid
    act[..., M_GREEN_LIGHT] = on_ego_path & tl_green & elem_valid

    R_ = route_lanes.shape[1]
    if R_ > 0 and L > 0:
        rl = route_lanes[..., :2]; ml = map_lanes[..., :2]
        Pq = min(rl.shape[2], ml.shape[2])
        diff = (rl[:, :, None, :Pq, :] - ml[:, None, :, :Pq, :]).abs().amax(dim=(3, 4))   # [B,R,L]
        dup = (diff < 0.5).any(-1)                                                       # [B,R]
        act[:, L + C:L + C + R_] &= ~dup.unsqueeze(-1)
    act = act & elem_valid.unsqueeze(-1)
    if return_aux:
        return act, {"stop_hit": stop_hit & act[..., M_STOP_LINE].unsqueeze(-1)}
    return act

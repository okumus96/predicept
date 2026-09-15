import argparse, collections, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_l1_labels import intent_labels, yields_to_ego, keeps_lane, DT, STOPPED, PERSIST
from Predicept.channels import (CH_FOLLOWS, CH_MERGES, CH_OVERTAKES, CH_COLLISION_COURSE,
                                 EV_DS_ENTRY, EV_DTHETA_FLOW, SAME_FLOW_RAD, LANE_W,
                                 _corridor_arrays, _project, compute_channels)
import torch, math

def gt_corridor(d, step_m=1.0, extend_m=100.0, P=1200):
    ef = d['ego_agent_future']
    pts = np.concatenate([np.array([[1e-3, 0.0]]), ef[:, :2]])
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= step_m: keep.append(i)
    pts = pts[keep]
    if len(pts) >= 2:
        h = math.atan2(*(pts[-1] - pts[-2])[::-1])
    else:
        h = 0.0
    n_ext = int(extend_m / step_m)
    ext = pts[-1][None] + np.arange(1, n_ext + 1)[:, None] * step_m * np.array([math.cos(h), math.sin(h)])
    pts = np.concatenate([pts, ext])
    yaw = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
    arr = np.zeros((P, 3), np.float32); n = min(P, len(pts))
    arr[:n, :2] = pts[:n]; arr[:n, 2] = yaw[:n]
    return torch.from_numpy(arr)[None, None]


def lane_corridor(d, step_m=1.0, extend_m=100.0, P=1200, max_hops=4):
    lanes = d['lanes']
    def valid_pts(k):
        pl = lanes[k]; v = np.abs(pl[:, :2]).sum(-1) > 1e-6
        pts = pl[v, :2]
        if len(pts) >= 2:
            hd = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
            cl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            i0 = int(np.searchsorted(cl, 3.0)); i0 = min(max(i0, 1), len(pts) - 1)
            iL = int(np.searchsorted(cl, cl[-1] - 3.0)); iL = max(min(iL, len(pts) - 2), 0)
            hd[0] = math.atan2(*(pts[i0] - pts[0])[::-1])
            hd[-1] = math.atan2(*(pts[-1] - pts[iL])[::-1])
        else:
            hd = pl[v, 2]
        return pts, hd
    def seg_dist(P_, pts):
        A, B = pts[:-1], pts[1:]; dv = B - A; L2 = (dv ** 2).sum(-1) + 1e-9
        t = np.clip(((P_ - A) * dv).sum(-1) / L2, 0, 1); C = A + t[:, None] * dv
        dd = np.linalg.norm(P_ - C, axis=1); i = int(dd.argmin()); return float(dd[i]), i
    ego_gt0 = d['ego_agent_future'][:, :2]
    cands = []
    for k in range(lanes.shape[0]):
        pts, hd = valid_pts(k)
        if len(pts) < 2: continue
        dd, i = seg_dist(np.zeros(2), pts)
        if dd <= 0.5 * LANE_W + 0.5 and abs(math.atan2(math.sin(hd[i]), math.cos(hd[i]))) <= math.radians(30.0):
            cands.append((dd, k))
    if not cands:
        return None
    ego_gt = d['ego_agent_future'][:, :2]

    def build_chain(k0):
        chain = [k0]; pts, hd = valid_pts(k0); poly = [pts]
        for _ in range(max_hops):
            end, end_hd = poly[-1][-1], hd[-1]
            cc = []
            for k in range(lanes.shape[0]):
                if k in chain: continue
                p2, h2 = valid_pts(k)
                if len(p2) < 2: continue
                dd = float(np.linalg.norm(p2[0] - end))
                if dd <= 1.5 and abs(math.atan2(math.sin(h2[0] - end_hd), math.cos(h2[0] - end_hd))) <= math.radians(45.0):
                    cc.append(k)
            if not cc: break
            if len(cc) == 1:
                nxt = cc[0]
            else:
                near = ego_gt[np.linalg.norm(ego_gt - end, axis=1) < 60.0]
                if len(near) >= 3:
                    def gt_dist(k):
                        p2, _ = valid_pts(k)
                        return float(np.linalg.norm(near[:, None, :] - p2[None, :, :], axis=-1).min(1).mean())
                    nxt = min(cc, key=gt_dist)
                else:
                    def hd_diff(k):
                        _, h2 = valid_pts(k)
                        return abs(math.atan2(math.sin(h2[0] - end_hd), math.cos(h2[0] - end_hd)))
                    nxt = min(cc, key=hd_diff)
            chain.append(nxt); p2, hd = valid_pts(nxt); poly.append(p2)
        return chain, poly

    def chain_score(poly):
        P = np.concatenate(poly); A, B = P[:-1], P[1:]; dv = B - A; L2 = (dv ** 2).sum(-1) + 1e-9
        t = np.clip(((ego_gt[:, None, :] - A[None]) * dv[None]).sum(-1) / L2[None], 0, 1)
        C = A[None] + t[..., None] * dv[None]
        return float(np.linalg.norm(ego_gt[:, None, :] - C, axis=-1).min(1).mean())

    if len(cands) == 1:
        chain, poly = build_chain(cands[0][1])
    else:
        best_sc, chain, poly = None, None, None
        for dd, k in sorted(cands):
            ch, po = build_chain(k); sc = chain_score(po)
            if best_sc is None or sc < best_sc - 1e-6:
                best_sc, chain, poly = sc, ch, po
    best = chain[0]
    pts = np.concatenate(poly)
    _, i0 = seg_dist(np.zeros(2), pts)
    A_, B_ = pts[i0], pts[i0 + 1]; AB_ = B_ - A_
    t_ = float(np.clip(-(A_ @ AB_) / max(float(AB_ @ AB_), 1e-9), 0.0, 1.0))
    pts = np.concatenate([(A_ + t_ * AB_)[None], pts[i0 + 1:]])
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= step_m: keep.append(i)
    pts = pts[keep]
    h = math.atan2(*(pts[-1] - pts[-2])[::-1]) if len(pts) >= 2 else 0.0
    n_ext = int(extend_m / step_m)
    ext = pts[-1][None] + np.arange(1, n_ext + 1)[:, None] * step_m * np.array([math.cos(h), math.sin(h)])
    pts = np.concatenate([pts, ext])
    if len(pts) < 2:
        return None
    yaw = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
    arr = np.zeros((P, 3), np.float32); n = min(P, len(pts))
    arr[:n, :2] = pts[:n]; arr[:n, 2] = yaw[:n]
    return torch.from_numpy(arr)[None, None]


MP_CLASSES = ['none', 'stopsAtTrafficControl', 'stopsForCrosswalk']
LC_INTO_MIN_M = 1.5
LC_INTO_SIDE_M = 2.3


def _pt_polyline_dist(P, pts):
    A, B = pts[:-1], pts[1:]; dv = B - A; L2 = (dv ** 2).sum(-1) + 1e-9
    t = np.clip(((P - A) * dv).sum(-1) / L2, 0, 1); C = A + t[:, None] * dv
    return float(np.linalg.norm(P - C, axis=1).min())


def changes_lane_into(d):
    lc = lane_corridor(d)
    if lc is None:
        return None
    lc_real = lane_corridor(d, extend_m=0.0)
    if lc_real is None:
        return None
    real_len = float((np.abs(lc_real[0, 0, :, :2].numpy()).sum(-1) > 1e-6).sum() - 1)
    cxy, cyaw, ccum, cv = _corridor_arrays(lc)
    pr = lambda P: [x[0].numpy() for x in _project(torch.from_numpy(np.ascontiguousarray(P)).float()[None], cxy, cyaw, ccum, cv)[:2]]
    _, el_f = pr(d['ego_agent_future'][:, :2]); _, el_p = pr(d['ego_agent_past'][:, :2])
    lat0 = float(np.median(el_p[-5:])); latE = float(np.median(el_f[-5:]))
    if abs(latE - lat0) < LC_INTO_MIN_M or abs(latE) < LC_INTO_SIDE_M:
        return None
    ef = d['ego_agent_future']
    hd_chg = abs(math.atan2(math.sin(ef[-1, 2] - ef[0, 2]), math.cos(ef[-1, 2] - ef[0, 2])))
    if hd_chg > math.radians(30.0):
        return None
    sgn = 1.0 if latE > lat0 else -1.0
    crossed = np.flatnonzero((el_f - lat0) * sgn >= 0.5 * LANE_W + 0.25)
    if len(crossed) == 0:
        return None
    t_star = int(crossed[0])
    settled = np.flatnonzero((el_f - lat0) * sgn >= LANE_W - 0.25)
    t_set = int(settled[0]) if len(settled) else len(ef) - 1
    t_set = max(t_set, t_star)
    P = ef[t_set, :2]
    a, b = max(t_set - 3, 0), min(t_set + 3, len(ef) - 1)
    hd_e = math.atan2(*(ef[b, :2] - ef[a, :2])[::-1]) if b > a else float(ef[t_set, 2])
    lanes = d['lanes']; best, bd = None, 0.5 * LANE_W
    for k in range(lanes.shape[0]):
        pl = lanes[k]; v = np.abs(pl[:, :2]).sum(-1) > 1e-6; pts = pl[v, :2]
        if len(pts) < 2: continue
        dd = _pt_polyline_dist(P, pts)
        if dd > bd: continue
        i = int(np.argmin(np.linalg.norm(pts - P, axis=1)))
        hd = math.atan2(*(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])[::-1])
        if abs(math.atan2(math.sin(hd - hd_e), math.cos(hd - hd_e))) > math.radians(30): continue
        fs_k, fl_k = pr(pts)
        if np.median(np.abs(fl_k)) < LC_INTO_SIDE_M: continue
        if _pt_polyline_dist(np.zeros(2), pts) < 0.5 * LANE_W + 0.5: continue
        past = d['ego_agent_past'][-10:, :2]
        if min(_pt_polyline_dist(q, pts) for q in past) < 0.5 * LANE_W + 0.5: continue
        par = (fs_k >= 0) & (fs_k <= real_len) & (np.abs(fl_k) >= LC_INTO_SIDE_M) & (np.abs(fl_k) <= 1.6 * LANE_W)
        if par.sum() < 2 or (fs_k[par].max() - fs_k[par].min()) < 10.0: continue
        bd, best = dd, k
    return best
TL_STOP_TOL = 3.0
STOP_BEFORE_M = 12.0


KL_LAT_TOL = 0.5 * LANE_W
KL_MIN_PTS = 5
KL_INSIDE_FRAC = 0.9


def keeps_lane_v3(ego_xy, lanes):
    out = np.zeros(lanes.shape[0], dtype=bool)
    for i in range(lanes.shape[0]):
        pl = lanes[i, :, :2]; v = np.abs(pl).sum(-1) > 1e-6
        if v.sum() < 2: continue
        P = pl[v]; A, B = P[:-1], P[1:]; AB = B - A
        L2 = np.maximum((AB * AB).sum(-1), 1e-9)
        t = ((ego_xy[:, None, :] - A[None]) * AB[None]).sum(-1) / L2[None]      # [T,S]
        tc = np.clip(t, 0.0, 1.0)
        C = A[None] + tc[..., None] * AB[None]
        dist = np.linalg.norm(ego_xy[:, None, :] - C, axis=-1)                   # [T,S]
        j = dist.argmin(1); dmin = dist[np.arange(len(ego_xy)), j]; tj = t[np.arange(len(ego_xy)), j]
        interior = ~(((j == 0) & (tj < 0.0)) | ((j == len(AB) - 1) & (tj > 1.0)))
        if interior.sum() < KL_MIN_PTS: continue
        inside = dmin[interior] <= KL_LAT_TOL
        out[i] = bool(inside.mean() >= KL_INSIDE_FRAC and inside[-KL_MIN_PTS:].all())
    return out


LC_PERSIST = 10          # 1.0 s
SAME_FLOW_LC = 0.55
MERGE_GAP_MAX = 40.0     # KG 0 < g <= 40 m
MERGE_HEADWAY_MAX = 5.0
OT_SIDE_LAT = 1.25
OT_GAP = 1.0
OT_V_S, OT_V_O = 2.0, 0.5
OT_STABLE = 5            # 0.5 s
OT_IN_LANE_M = 1.2
LC_IN_LANE_M = 1.2
LC_ADJ_MIN_M = 2.3


def _band(lat):
    return np.rint(lat / LANE_W).astype(int)


def _persistent_final_band(band, n=LC_PERSIST):
    if len(band) < n: return None
    tail = band[-n:]
    return int(tail[0]) if (tail == tail[0]).all() else None


def lane_change_labels(d, cxy, cyaw, ccum, cv):
    N = 10
    ego_fut = d['ego_agent_future'][:, :2]; ego_past = d['ego_agent_past'][:, :2]
    egoL, egoW = 4.62, 2.10
    pr = lambda P: [x[0].numpy() for x in _project(torch.from_numpy(np.ascontiguousarray(P)).float()[None], cxy, cyaw, ccum, cv)[:2]]
    es_f, el_f = pr(ego_fut); es_p, el_p = pr(ego_past)
    v_ego = np.linalg.norm(np.diff(ego_fut, axis=0), axis=-1) / DT; v_ego = np.append(v_ego, v_ego[-1])
    eb_f = _band(el_f); ego_T = _persistent_final_band(eb_f)
    ego_b0 = int(np.round(np.median(el_p[-5:]) / LANE_W)) if len(el_p) >= 5 else 0
    ego_changes = (ego_T is not None and ego_T != ego_b0
                   and abs(float(el_f[-1]) - float(np.median(el_p[-5:]))) >= 0.7 * LANE_W)
    ego_hd = np.arctan2(np.gradient(ego_fut[:, 1]), np.gradient(ego_fut[:, 0]))
    cut = np.zeros(N, bool); ovt = np.zeros(N, bool); mrg = np.zeros(N, bool)
    A = []
    for j in range(N):
        cur = d['neighbor_agents_past'][j, -1]
        if np.abs(cur[:2]).sum() < 1e-6 or int(cur[8:11].argmax()) != 0: continue
        fut = d['neighbor_agents_future'][j]; ok = np.abs(fut[:, :2]).sum(-1) > 1e-6
        if ok.sum() < 20: continue
        past = d['neighbor_agents_past'][j]; okp = np.abs(past[:, :2]).sum(-1) > 1e-6
        as_f, al_f = pr(fut[ok, :2]); as_p, al_p = pr(past[okp, :2])
        ab_f, ab_p = _band(al_f), _band(al_p)
        T = ok.sum()
        a_hd = np.arctan2(np.gradient(fut[ok, 1]), np.gradient(fut[ok, 0]))
        dth = np.abs(np.arctan2(np.sin(a_hd - ego_hd[:T]), np.cos(a_hd - ego_hd[:T])))
        same_flow_frac = float((dth <= SAME_FLOW_LC).mean())
        L_a = max(float(cur[6]), 1.0)
        gap_ahead = as_f - es_f[:T] - 0.5 * (L_a + egoL)
        v_a = np.linalg.norm(np.diff(fut[ok, :2], axis=0), axis=-1) / DT; v_a = np.append(v_a, v_a[-1])
        A.append((j, ab_p, ab_f, as_f, gap_ahead, same_flow_frac, v_a, T, al_p, al_f))
    ego_veh_len = egoL
    for (j, ab_p, ab_f, as_f, gap_ahead, sff, v_a, T, al_p, al_f) in A:
        a_T = _persistent_final_band(ab_f)
        ego_lat_p = float(np.median(el_p[-5:])) if len(el_p) >= 5 else 0.0
        rel_p = al_p - ego_lat_p
        rel_f = al_f - el_f[:T]
        ego_moved = abs(float(el_f[-1]) - ego_lat_p)
        agent_moved = float(np.abs(al_f - al_f[0]).max())
        rel0 = float(np.median(rel_p)) if rel_p.size else float(rel_f[0])
        end_near_ego = (np.abs(rel_f[-LC_PERSIST:]) <= LC_IN_LANE_M).all() if T >= LC_PERSIST else False
        end_in_ego_lane = end_near_ego and (np.abs(al_f[-LC_PERSIST:]) <= LC_IN_LANE_M).all()
        g = gap_ahead[-LC_PERSIST:]
        vt = v_ego[T - 1] if T <= len(v_ego) else v_ego[-1]
        gap_ok = (g > 0).all() and g[-1] <= MERGE_GAP_MAX and (vt <= 0.30 or g[-1] / vt <= MERGE_HEADWAY_MAX)
        if (abs(rel0) >= LC_ADJ_MIN_M and end_in_ego_lane and agent_moved >= 1.5 and agent_moved > ego_moved
                and sff >= 0.8 and gap_ok):
            cut[j] = True
        if (abs(rel0) >= LC_ADJ_MIN_M and end_near_ego and ego_moved >= 1.5 and ego_moved >= agent_moved
                and agent_moved < 1.0 and sff >= 0.8 and gap_ok):
            mrg[j] = True
        if (not ego_changes) and sff >= 0.8:
            ego_lat_p = float(np.median(el_p[-5:])) if len(el_p) >= 5 else 0.0
            rel_p = al_p - ego_lat_p
            rel_f = al_f - el_f[:T]
            behind0 = (np.abs(rel_p) <= OT_IN_LANE_M).any() and (gap_ahead[0] < -OT_GAP or (as_p[-1] - es_p[-1]) < -OT_GAP)
            side = np.abs(rel_f) >= 0.5 * LANE_W + OT_SIDE_LAT
            if behind0 and side.any():
                t_side = int(np.argmax(side))
                after = np.arange(T) > t_side
                ahead = after & (gap_ahead >= OT_GAP)
                back = after & (np.abs(rel_f) <= OT_IN_LANE_M) & (gap_ahead >= OT_GAP)
                stable = back[-OT_STABLE:].all() if T >= OT_STABLE else False
                v_ok = (v_a.max() >= OT_V_S) and (v_ego[:T].max() >= OT_V_O)
                if ahead.any() and stable and v_ok:
                    ovt[j] = True
    if cut.any():
        cands = [(gap_ahead[-1], j) for (j, ab_p, ab_f, as_f, gap_ahead, sff, v_a, T, _, _) in A if cut[j]]
        keep = min(cands)[1]; cut[:] = False; cut[keep] = True
    if mrg.any():
        cands = [(gap_ahead[-1], j) for (j, ab_p, ab_f, as_f, gap_ahead, sff, v_a, T, _, _) in A if mrg[j]]
        keep = min(cands)[1]; mrg[:] = False; mrg[keep] = True
    return cut, ovt, mrg


_MAPS = {}
def _stop_sign_mask(d):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.maps.abstract_map import SemanticMapLayer as SL
    mname = str(d['map_name'])
    if mname not in _MAPS:
        _MAPS[mname] = get_maps_api(os.environ.get('NUPLAN_MAPS_ROOT', 'nuplan/dataset/maps'), 'nuplan-maps-v1.0', mname)
    api = _MAPS[mname]
    loc, glo = d['c_lat_candidates'][0], d['c_lat_candidates_global'][0]
    th = float(glo[0, 2] - loc[0, 2]); c, s_ = math.cos(th), math.sin(th)
    ex, ey = glo[0, :2] - np.array([loc[0, 0] * c - loc[0, 1] * s_, loc[0, 0] * s_ + loc[0, 1] * c])
    ce, se = math.cos(-th), math.sin(-th)
    to_ego = lambda P: np.stack([(P[:, 0] - ex) * ce - (P[:, 1] - ey) * se, (P[:, 0] - ex) * se + (P[:, 1] - ey) * ce], -1)
    stops = api.get_proximal_map_objects(Point2D(float(ex), float(ey)), 120.0, [SL.STOP_LINE])[SL.STOP_LINE]
    signs = [to_ego(np.array(o.polygon.exterior.coords)).mean(0) for o in stops if int(o.stop_line_type) == 1]
    out = np.zeros(d['stop_polygons'].shape[0], bool)
    for q in range(len(out)):
        P = d['stop_polygons'][q]; pv = np.abs(P[:, :2]).sum(-1) > 1e-6
        if pv.sum() < 3 or not signs: continue
        cq = P[pv, :2].mean(0); out[q] = any(np.linalg.norm(sg - cq) < 6.0 for sg in signs)
    return out


def map_labels(d):
    lanes = d['lanes']; L = lanes.shape[0]
    nS = L + d['crosswalks'].shape[0] + d['route_lanes'].shape[0]
    mp = np.zeros(nS, np.int64)
    ego_xy = np.concatenate([d['ego_agent_past'][-1:, :2], d['ego_agent_future'][:, :2]])
    ego_v = np.linalg.norm(ego_xy[1:] - ego_xy[:-1], axis=-1) / DT
    stopped = ego_v <= STOPPED
    if not stopped[35:40].all():
        return mp
    stop_idx = 39
    ref = gt_corridor(d)
    cxy, cyaw, ccum, cv = _corridor_arrays(ref)
    s_stop = float(_project(torch.from_numpy(ego_xy[stop_idx + 1:stop_idx + 2]).float()[None],
                            cxy, cyaw, ccum, cv)[0][0, 0])
    lxy = lanes[..., :2]; lv = np.abs(lxy).sum(-1) > 1e-6
    fs, fl, ftan, _ = _project(torch.from_numpy(lxy.reshape(1, -1, 2)).float(), cxy, cyaw, ccum, cv)
    fs, fl, ftan = fs[0].numpy().reshape(L, -1), fl[0].numpy().reshape(L, -1), ftan[0].numpy().reshape(L, -1)
    dth = np.abs(np.arctan2(np.sin(lanes[..., 2] - ftan), np.cos(lanes[..., 2] - ftan)))
    on_pt = (np.abs(fl) <= 0.5 * LANE_W) & (fs > 0) & (fs <= 80) & lv & (dth <= SAME_FLOW_RAD)
    frac = on_pt.sum(-1) / np.maximum(lv.sum(-1), 1)
    on_path = on_pt & (frac >= 0.5)[:, None]
    red_k = (d['lane_tl'][..., 2] > 0).any(-1)
    sign_q = _stop_sign_mask(d)
    sp = d['stop_polygons'][sign_q][..., :2].reshape(-1, 2) if sign_q.any() else np.zeros((0, 2)); spv = np.abs(sp).sum(-1) > 1e-6
    P_stop = ego_xy[stop_idx + 1]
    any_hit = False; fb_best, fb_d = None, 0.5 * LANE_W + 0.5
    for k in range(L):
        if lv[k].sum() < 2: continue
        near_sign = spv.any() and (np.linalg.norm(lxy[k][lv[k]][:, None, :] - sp[spv][None], axis=-1).min() <= TL_STOP_TOL)
        if not (red_k[k] or near_sign): continue
        if on_path[k].any():
            s_lo, s_hi = fs[k][on_path[k]].min(), fs[k][on_path[k]].max()
            if s_lo - STOP_BEFORE_M <= s_stop <= s_hi:
                mp[k] = MP_CLASSES.index('stopsAtTrafficControl'); any_hit = True; continue
        dd = _pt_polyline_dist(P_stop, lxy[k][lv[k]])
        if dd < fb_d: fb_d, fb_best = dd, k
    if not any_hit and fb_best is not None:
        mp[fb_best] = MP_CLASSES.index('stopsAtTrafficControl')
    if not (mp[:L] == MP_CLASSES.index('stopsAtTrafficControl')).any():
        C = d['crosswalks'].shape[0]
        for c in range(C):
            Q = d['crosswalks'][c][:, :2]; qv = np.abs(Q).sum(-1) > 1e-6
            if qv.sum() < 3: continue
            qs, ql, _, _ = _project(torch.from_numpy(Q[qv]).float()[None], cxy, cyaw, ccum, cv)
            qs, ql = qs[0].numpy(), ql[0].numpy()
            onp = (np.abs(ql) <= 0.5 * LANE_W) & (qs > 0) & (qs <= 80)
            if not onp.any(): continue
            s_cw = float(qs[onp].min())
            nb = d['neighbor_agents_past'][:, -1]; vru = (nb[:, 8:11].argmax(-1) >= 1) & (np.abs(nb[:, :2]).sum(-1) > 1e-6)
            vru_near = vru.any() and (np.linalg.norm(nb[vru][:, None, :2] - Q[qv][None], axis=-1).min() <= 5.0)
            if vru_near and (s_cw - STOP_BEFORE_M <= s_stop <= s_cw + 2.0):
                mp[L + c] = MP_CLASSES.index('stopsForCrosswalk')
    return mp


AG_CLASSES = ['none', 'follows', 'givesWayTo', 'cutsInAhead', 'mergesBehind']
AG_PRIORITY = ['givesWayTo', 'cutsInAhead', 'mergesBehind', 'follows']
EGO_L, EGO_W = 4.62, 2.10
COL_HORIZON = 80          # 8 s @ 10 Hz


def _corners(cx, cy, hd, L, W):
    c, s = np.cos(hd), np.sin(hd)
    dx = np.stack([ L/2,  L/2, -L/2, -L/2], -1)[None] * np.ones_like(cx)[:, None]
    dy = np.stack([ W/2, -W/2, -W/2,  W/2], -1)[None] * np.ones_like(cx)[:, None]
    x = cx[:, None] + dx * c[:, None] - dy * s[:, None]
    y = cy[:, None] + dx * s[:, None] + dy * c[:, None]
    return np.stack([x, y], -1)


def _sat_overlap(A, B):
    for P in (A, B):
        for i in range(4):
            e = P[(i+1) % 4] - P[i]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def observed_collision(ego_fut, nb_last, nb_fut):
    N = nb_fut.shape[0]
    T = min(COL_HORIZON, ego_fut.shape[0])
    eC = _corners(ego_fut[:T, 0], ego_fut[:T, 1], ego_fut[:T, 2], EGO_L, EGO_W)   # [T,4,2]
    col = np.zeros(N, bool); ttc = np.full(N, np.inf)
    for j in range(N):
        if np.abs(nb_last[j, :2]).sum() < 1e-6: continue
        ok = np.abs(nb_fut[j, :T, :2]).sum(-1) > 1e-6
        if ok.sum() < 5: continue
        L, W = max(float(nb_last[j, 6]), 1.0), max(float(nb_last[j, 7]), 0.6)
        aC = _corners(nb_fut[j, :T, 0], nb_fut[j, :T, 1], nb_fut[j, :T, 2], L, W)
        d = np.linalg.norm(nb_fut[j, :T, :2] - ego_fut[:T, :2], axis=-1)
        cand = np.where(ok & (d <= (np.hypot(EGO_L, EGO_W) + np.hypot(L, W)) / 2))[0]
        for t in cand:
            if _sat_overlap(eC[t], aC[t]):
                col[j] = True; ttc[j] = (t + 1) * DT; break
    return col, ttc


def main(a):
    files = sorted(glob.glob(a.valid_set + "/*.npz"))
    if a.shard:
        k, n = map(int, a.shard.split("/")); files = files[k::n]
    print(f"[data] {len(files)} scenes" + (f" (shard {a.shard})" if a.shard else ""))
    AG, CR, CG, CT, FILES, MP = [], [], [], [], [], []
    RAW_NAMES = ['yieldingTo', 'waitingFor', 'cutsInAhead', 'mergesBehind', 'follows',
                 'col_refpath', 'col_gt']
    RAW, VALID = [], []
    multi = collections.Counter(); cnt = collections.Counter()
    failed = []
    for f in files:
      try:
        d = np.load(f)
        N = 10
        ego_xy = np.concatenate([d['ego_agent_past'][-1:, :2], d['ego_agent_future'][:, :2]])
        ego_v = np.concatenate([[0.], np.linalg.norm(ego_xy[1:] - ego_xy[:-1], axis=-1) / DT]); ego_v[0] = ego_v[1]
        nb_last, nb_fut = d['neighbor_agents_past'][:N, -1], d['neighbor_agents_future'][:N]
        if 'channel_active_gt' in d.files:
            gt_ref, ev_ref = d['channel_active_gt'][:N], d['channel_evidence_gt'][:N]
        else:
            a_rp, e_rp = compute_channels(torch.from_numpy(d['neighbor_agents_past'][:N]).float()[None],
                                          torch.from_numpy(d['ego_agent_past']).float()[None],
                                          torch.from_numpy(nb_fut[:, :, :2]).float()[None],
                                          torch.from_numpy(d['c_lat_candidates']).float()[None])
            gt_ref, ev_ref = a_rp[0].numpy(), e_rp[0].numpy()
        a_gt, e_gt = compute_channels(torch.from_numpy(d['neighbor_agents_past'][:N]).float()[None],
                                      torch.from_numpy(d['ego_agent_past']).float()[None],
                                      torch.from_numpy(nb_fut[:, :, :2]).float()[None], gt_corridor(d))
        gt, ev = a_gt[0].numpy(), e_gt[0].numpy()
        efp = d['ego_agent_future'][:, :2]; L_gt = float(np.linalg.norm(efp[1:] - efp[:-1], axis=-1).sum())
        cxy_, cyaw_, ccum_, cv_ = _corridor_arrays(gt_corridor(d))
        agent_in_lane_fs = np.full(N, np.inf)
        for jj in range(N):
            okf = np.abs(nb_fut[jj, :, :2]).sum(-1) > 1e-6
            if okf.sum() < 2: continue
            fs_, fl_, _, _ = _project(torch.from_numpy(nb_fut[jj, okf, :2]).float()[None], cxy_, cyaw_, ccum_, cv_)
            fs_, fl_ = fs_[0].numpy(), fl_[0].numpy()
            m_ = (np.abs(fl_) <= 0.5 * LANE_W) & (fs_ > 0)
            if m_.any(): agent_in_lane_fs[jj] = float(fs_[m_].min())
        ys, ws = intent_labels(ego_xy, ego_v, nb_last, nb_fut)
        yte = yields_to_ego(ego_xy, nb_last, nb_fut)
        lc = lane_corridor(d)
        if lc is None:
            cut_j = np.zeros(N, bool); ovt_j = np.zeros(N, bool); mrg_j = np.zeros(N, bool)
        else:
            rxy, ryaw, rcum, rcv = _corridor_arrays(lc)
            cut_j, ovt_j, mrg_j = lane_change_labels(d, rxy, ryaw, rcum, rcv)
        ag = np.zeros(N, np.int64)
        raw = np.zeros((N, len(RAW_NAMES)), bool)
        valid = np.abs(d['neighbor_agents_past'][:N]).sum(axis=(1, 2)) > 0
        for j in range(N):
            if np.abs(d['neighbor_agents_past'][j]).sum() == 0: continue
            fired = []
            if j in ys or j in ws: fired.append('givesWayTo')
            if cut_j[j]: fired.append('cutsInAhead')
            if mrg_j[j]: fired.append('mergesBehind')
            if gt[j, CH_FOLLOWS]: fired.append('follows')
            if len(fired) > 1: multi[tuple(sorted(fired))] += 1
            for c in fired:
                if c == 'givesWayTo':
                    raw[j, RAW_NAMES.index('yieldingTo')] = j in ys; raw[j, RAW_NAMES.index('waitingFor')] = j in ws
                else: raw[j, RAW_NAMES.index(c)] = True
            for c in AG_PRIORITY:
                if c in fired: ag[j] = AG_CLASSES.index(c); break
            cnt[AG_CLASSES[ag[j]]] += 1
        col_g, ttc_g = observed_collision(d['ego_agent_future'], nb_last, nb_fut)
        raw[:, RAW_NAMES.index('col_refpath')] = gt_ref[:, CH_COLLISION_COURSE].astype(bool) & valid
        raw[:, RAW_NAMES.index('col_gt')] = np.asarray(col_g, bool) & valid
        RAW.append(raw); VALID.append(valid)
        AG.append(ag); CR.append(gt_ref[:, CH_COLLISION_COURSE].astype(bool)); CG.append(col_g); CT.append(ttc_g)
        MP.append(map_labels(d))
        FILES.append(os.path.basename(f))
      except Exception as e:
        failed.append(os.path.basename(f)); print(f'[SKIP] {os.path.basename(f)}: {type(e).__name__}: {e}')
    print(f'[data] {len(FILES)} scenes processed, {len(failed)} skipped')
    AG, CR, CG, CT, MP, RAW, VALID = map(np.stack, (AG, CR, CG, CT, MP, RAW, VALID))
    np.savez(a.out, agent=AG, col_refpath=CR, col_gt=CG, col_gt_ttc=CT, map=MP,
             files=np.array(FILES), ag_classes=np.array(AG_CLASSES), mp_classes=np.array(MP_CLASSES),
             raw=RAW, raw_names=np.array(RAW_NAMES), valid=VALID)
    print("\nMAP CLASS DISTRIBUTION (elements):")
    for c in MP_CLASSES: print(f"   {c:22s} {int((MP == MP_CLASSES.index(c)).sum()):7d}")
    print(f"   scenes with stopsAtTrafficControl: {int((MP == MP_CLASSES.index('stopsAtTrafficControl')).any(1).sum())} / {len(FILES)}")
    n = int((AG >= 0).sum())
    print("\nCLASS DISTRIBUTION (10 agents x scenes):")
    for c in AG_CLASSES: print(f"   {c:12s} {cnt[c]:7d}")
    print(f"\nagents with >=2 overlapping classes: {sum(multi.values())}  " + str(dict(multi.most_common(4))))
    print(f"\nCOLLISION:  col_refpath {int(CR.sum())}   col_gt {int(CG.sum())}   BOTH {int((CR & CG).sum())}"
          f"   refpath only {int((CR & ~CG).sum())}   gt only {int((~CR & CG).sum())}")
    v = CT[np.isfinite(CT)]
    if len(v): print(f"col_gt first collision time: p25 {np.percentile(v,25):.1f}  p50 {np.percentile(v,50):.1f}  p75 {np.percentile(v,75):.1f} s")
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--valid_set", required=True)
    p.add_argument("--out", default="results_label_identity/l1_labels_v3.npz")
    p.add_argument("--shard", default="", help="k/n: k-th shard of the file list (0-based)")
    main(p.parse_args())

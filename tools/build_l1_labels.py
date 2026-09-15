import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Predicept.channels import CH_FOLLOWS, CH_MERGES, CH_OVERTAKES

DT = 0.1
YIELD_HOR = 60
STOPPED = 0.30
DECEL = 0.30
ARRIVAL_TOL = 1.5
PERSIST = 10
WAIT_BLOCK_S = 3.0
CROSS_MIN_RAD = 0.52
EGO_W = 2.297
LANE_LAT_TOL = 1.75
LANE_COVER = 0.5

AG_CLASSES = ['none', 'np:follows', 'np:yieldingTo', 'np:waitingFor',
              'np:mergesInFrontOf', 'np:overtakes']
MP_CLASSES = ['none', 'np:keepsLane']
AG_PRIORITY = ['np:yieldingTo', 'np:waitingFor', 'np:mergesInFrontOf',
               'np:overtakes', 'np:follows']


def _arrival(path, pt, rad):
    d = np.linalg.norm(path - pt, axis=-1)
    m = d <= rad
    return float(np.argmax(m)) * DT if m.any() else np.inf


def _heading(path, i):
    a = max(0, i - 3); b = min(len(path) - 1, i + 3)
    d = path[b] - path[a]
    return np.arctan2(d[1], d[0])


def intent_labels(ego_xy, ego_v, nb_last, nb_fut):
    ys, ws = set(), set()
    for j in range(nb_last.shape[0]):
        if np.abs(nb_last[j]).sum() == 0:
            continue
        obj = np.concatenate([nb_last[j:j + 1, :2], nb_fut[j, :, :2]])
        ovalid = np.abs(obj).sum(-1) > 1e-6
        if ovalid.sum() < YIELD_HOR // 2:
            continue
        obj = np.where(ovalid[:, None], obj, 1e4)
        W = max(float(nb_last[j, 7]), 1.0)
        thr, rad = (EGO_W + W) / 2.0, max(EGO_W, W)
        y_streak = y_best = w_streak = w_best = 0
        DF = np.linalg.norm(ego_xy[:, None, :] - obj[None, :, :], axis=-1)
        for k in range(PERSIST + 1):
            sp, op = ego_xy[k:k + YIELD_HOR], obj[k:k + YIELD_HOR]
            if len(sp) < 5 or len(op) < 5:
                break
            D = DF[k:k + YIELD_HOR, k:k + YIELD_HOR]
            if D.min() > thr:
                y_streak = w_streak = 0
                continue
            a, b = np.unravel_index(D.argmin(), D.shape)
            dth = _heading(sp, a) - _heading(op, b)
            if abs(np.arctan2(np.sin(dth), np.cos(dth))) < CROSS_MIN_RAD:
                y_streak = w_streak = 0
                continue
            pt = 0.5 * (sp[a] + op[b])
            sa, oa = _arrival(sp, pt, rad), _arrival(op, pt, rad)
            acc = (ego_v[min(k + 5, len(ego_v) - 1)] - ego_v[k]) / 0.5
            decel, stopped = acc <= -DECEL, ego_v[k] <= STOPPED
            clears = np.isfinite(oa) and (not np.isfinite(sa) or oa + ARRIVAL_TOL < sa)
            y_streak = y_streak + 1 if ((decel and not stopped) and clears) else 0
            y_best = max(y_best, y_streak)
            blocks = np.isfinite(oa) and oa <= WAIT_BLOCK_S
            w_streak = w_streak + 1 if (stopped and blocks) else 0
            w_best = max(w_best, w_streak)
        if y_best >= PERSIST - 2:
            ys.add(j)
        if w_best >= PERSIST - 2:
            ws.add(j)
    return ys, ws


def yields_to_ego(ego_xy, nb_last, nb_fut):
    out = set()
    for j in range(nb_last.shape[0]):
        if np.abs(nb_last[j]).sum() == 0:
            continue
        if int(np.argmax(nb_last[j, 8:11])) == 1:
            continue
        obj = np.concatenate([nb_last[j:j + 1, :2], nb_fut[j, :, :2]])
        ovalid = np.abs(obj).sum(-1) > 1e-6
        if ovalid.sum() < YIELD_HOR // 2:
            continue
        obj = np.where(ovalid[:, None], obj, 1e4)
        ov = np.concatenate([[0.], np.linalg.norm(obj[1:] - obj[:-1], axis=-1) / DT]); ov[0] = ov[1]
        ov[~ovalid] = 0.0
        W = max(float(nb_last[j, 7]), 1.0)
        thr, rad = (EGO_W + W) / 2.0, max(EGO_W, W)
        DF = np.linalg.norm(ego_xy[:, None, :] - obj[None, :, :], axis=-1)
        streak = best = 0
        for k in range(PERSIST + 1):
            sp, op = ego_xy[k:k + YIELD_HOR], obj[k:k + YIELD_HOR]
            if len(sp) < 5 or len(op) < 5:
                break
            D = DF[k:k + YIELD_HOR, k:k + YIELD_HOR]
            if D.min() > thr:
                streak = 0
                continue
            a, b = np.unravel_index(D.argmin(), D.shape)
            dth = _heading(sp, a) - _heading(op, b)
            if abs(np.arctan2(np.sin(dth), np.cos(dth))) < CROSS_MIN_RAD:
                streak = 0
                continue
            pt = 0.5 * (sp[a] + op[b])
            ea, aa = _arrival(sp, pt, rad), _arrival(op, pt, rad)
            acc = (ov[min(k + 5, len(ov) - 1)] - ov[k]) / 0.5
            decel, stopped = acc <= -DECEL, ov[k] <= STOPPED
            ego_clears = np.isfinite(ea) and (not np.isfinite(aa) or ea + ARRIVAL_TOL < aa)
            streak = streak + 1 if ((decel or stopped) and ego_clears) else 0
            best = max(best, streak)
        if best >= PERSIST - 2:
            out.add(j)
    return out


def keeps_lane(ego_xy, lanes):
    out = np.zeros(lanes.shape[0], dtype=bool)
    for i in range(lanes.shape[0]):
        pl = lanes[i, :, :2]
        v = np.abs(pl).sum(-1) > 1e-6
        if v.sum() < 2:
            continue
        d = np.linalg.norm(ego_xy[:, None, :] - pl[None, v, :], axis=-1).min(1)
        out[i] = (d <= LANE_LAT_TOL).mean() >= LANE_COVER
    return out


def main(a):
    files = sorted(glob.glob(a.valid_set + "/*.npz"))
    print(f"[data] {len(files)} scenes")
    N, S = 10, None
    AG, MP, FILES = [], [], []
    multi = collections.Counter()
    for f in files:
        d = np.load(f)
        ego_xy = np.concatenate([d['ego_agent_past'][-1:, :2], d['ego_agent_future'][:, :2]])
        ego_v = np.concatenate([[0.], np.linalg.norm(ego_xy[1:] - ego_xy[:-1], axis=-1) / DT])
        ego_v[0] = ego_v[1]
        nb_last, nb_fut = d['neighbor_agents_past'][:N, -1], d['neighbor_agents_future'][:N]
        gt = d['channel_active_gt'][:N]
        ys, ws = intent_labels(ego_xy, ego_v, nb_last, nb_fut)

        ag = np.zeros(N, dtype=np.int64)
        for j in range(N):
            if np.abs(d['neighbor_agents_past'][j]).sum() == 0:
                continue
            fired = []
            if j in ys: fired.append('np:yieldingTo')
            if j in ws: fired.append('np:waitingFor')
            if gt[j, CH_MERGES]: fired.append('np:mergesInFrontOf')
            if gt[j, CH_OVERTAKES]: fired.append('np:overtakes')
            if gt[j, CH_FOLLOWS]: fired.append('np:follows')
            if len(fired) > 1:
                multi[tuple(sorted(fired))] += 1
            for c in AG_PRIORITY:
                if c in fired:
                    ag[j] = AG_CLASSES.index(c)
                    break
        lanes = d['lanes']
        kl = keeps_lane(ego_xy, lanes)
        nS = lanes.shape[0] + d['crosswalks'].shape[0] + d['route_lanes'].shape[0]
        S = nS if S is None else S
        mp = np.zeros(nS, dtype=np.int64)
        mp[:lanes.shape[0]] = kl.astype(np.int64)
        AG.append(ag); MP.append(mp); FILES.append(os.path.basename(f))

    AG, MP = np.stack(AG), np.stack(MP)
    print(f"\n=== AGENT L1 ({AG.size} agent slots, {int((AG >= 0).sum())} records) ===")
    valid = AG.reshape(-1)
    cnt = collections.Counter(valid.tolist())
    for k in range(len(AG_CLASSES)):
        print(f"  {AG_CLASSES[k]:<22s} {cnt[k]:7d}  %{100 * cnt[k] / len(valid):5.2f}")
    print(f"  scenes with at least one non-none agent: "
          f"{int((AG > 0).any(1).sum())}/{len(AG)}  (%{100 * (AG > 0).any(1).mean():.1f})")
    print(f"\n=== MAP L1 ({MP.shape[1]} elements) ===")
    m = MP.reshape(-1)
    for k in range(len(MP_CLASSES)):
        print(f"  {MP_CLASSES[k]:<22s} {int((m == k).sum()):7d}  %{100 * (m == k).mean():5.2f}")
    print(f"  keepsLane elements per scene: median "
          f"{np.median((MP == 1).sum(1)):.0f}, max {int((MP == 1).sum(1).max())}")
    print(f"  scenes without keepsLane: {int(((MP == 1).sum(1) == 0).sum())}/{len(MP)}")
    if multi:
        print(f"\n=== MULTI-LABEL (reduced to one label by priority) ===")
        for k, v in multi.most_common(8):
            print(f"  {' + '.join(x.replace('np:', '') for x in k):<48s} {v}")
    np.savez_compressed(a.out, agent=AG, map=MP, files=np.array(FILES),
                        ag_classes=np.array(AG_CLASSES), mp_classes=np.array(MP_CLASSES))
    print(f"\n[done] -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--valid_set", required=True)
    p.add_argument("--out", default="l1_labels.npz")
    main(p.parse_args())

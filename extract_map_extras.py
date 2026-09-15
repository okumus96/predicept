import argparse, glob, math, os, sys
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VERSION = "map-extras-v1.0"

def ego_pose(d):
    loc, glo = d['c_lat_candidates'][0], d['c_lat_candidates_global'][0]
    th = float(glo[0, 2] - loc[0, 2]); c, s = math.cos(th), math.sin(th)
    t = glo[0, :2] - np.array([loc[0, 0] * c - loc[0, 1] * s, loc[0, 0] * s + loc[0, 1] * c])
    return float(t[0]), float(t[1]), th

def main(a):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.maps.abstract_map import SemanticMapLayer as SL
    files = sorted(glob.glob(os.path.join(a.data, '*.npz')))
    if a.shard:
        k, n = map(int, a.shard.split('/')); files = files[k::n]
    if a.limit: files = files[:a.limit]
    print(f'{len(files)} files (apply={a.apply}, version={VERSION})', flush=True)
    maps = {}; done = skip = err = 0
    n_sl = np.zeros(7, int); lim_known = lim_tot = 0
    for f in tqdm(files, mininterval=30.0):
        try:
            d = dict(np.load(f, allow_pickle=True))
        except Exception as e:
            print(f'[ERR-READ] {f}: {e}', flush=True); err += 1; continue
        if (not a.force) and str(d.get('map_extras_version', '')) == VERSION:
            skip += 1; continue
        mname = str(d['map_name'])
        if mname not in maps:
            maps[mname] = get_maps_api(a.map_path, 'nuplan-maps-v1.0', mname)
        api = maps[mname]
        ex, ey, th = ego_pose(d); ce, se = math.cos(-th), math.sin(-th)
        to_ego = lambda P: np.stack([(P[:, 0] - ex) * ce - (P[:, 1] - ey) * se,
                                     (P[:, 0] - ex) * se + (P[:, 1] - ey) * ce], -1)
        prox = api.get_proximal_map_objects(Point2D(ex, ey), 120.0, [SL.LANE, SL.LANE_CONNECTOR, SL.STOP_LINE])
        lane_objs = [(o, to_ego(np.array([[p.x, p.y] for p in o.baseline_path.discrete_path])))
                     for lay in (SL.LANE, SL.LANE_CONNECTOR) for o in prox[lay]]
        L = d['lanes'].shape[0]
        lim = np.zeros(L, np.float32)
        for k in range(L):
            pl = d['lanes'][k][:, :2]; v = np.abs(pl).sum(-1) > 1e-6
            if v.sum() < 2: continue
            pts = pl[v]; best = None
            for o, B in lane_objs:
                if len(B) < 2: continue
                cover = float((np.linalg.norm(B[::2][:, None, :] - pts[None], axis=-1).min(1) < 1.5).mean())
                if cover >= 0.6:
                    sl_ = getattr(o, 'speed_limit_mps', None)
                    if sl_: best = float(sl_) if best is None else max(best, float(sl_))
            if best is not None: lim[k] = best
        lim_tot += int((np.abs(d['lanes'][..., :2]).sum(-1) > 1e-6).any(-1).sum()); lim_known += int((lim > 0).sum())
        sp = d['stop_polygons']; K = sp.shape[0]
        typ = np.full(K, -1, np.int8)
        live = [(o, to_ego(np.array(o.polygon.exterior.coords)).mean(0)) for o in prox[SL.STOP_LINE]]
        for q in range(K):
            P = sp[q][:, :2]; v = np.abs(P).sum(-1) > 1e-6
            if v.sum() < 3 or not live: continue
            c = P[v].mean(0)
            o, dd = min(((o, float(np.linalg.norm(cc - c))) for o, cc in live), key=lambda t: t[1])
            if dd <= 6.0: typ[q] = int(o.stop_line_type)
        for t_ in typ[typ >= 0]: n_sl[min(int(t_), 6)] += 1
        d['lane_speed_limit'] = lim; d['stop_line_type'] = typ; d['map_extras_version'] = VERSION
        if a.apply:
            tmp = f + '.tmp.npz'
            try:
                np.savez(tmp, **d); os.replace(tmp, f)
                if done == 0:
                    back = np.load(f, allow_pickle=True)
                    assert 'lane_speed_limit' in back.files and 'stop_line_type' in back.files
                    print(f'[VERIFY] first file: {len(back.files)} keys', flush=True)
            except Exception as e:
                if os.path.exists(tmp): os.remove(tmp)
                print(f'[ERR-WRITE] {f}: {e}', flush=True); err += 1; continue
        done += 1
    names = ['PED_CROSSING', 'STOP_SIGN', 'TRAFFIC_LIGHT', 'TURN_STOP', 'YIELD', 'UNKNOWN', '6+']
    print(f'\nDONE: processed={done} skipped={skip} errors={err} {"(DRY RUN)" if not a.apply else ""}')
    print(f'lanes with known speed limit: {lim_known}/{lim_tot} ({100*lim_known/max(lim_tot,1):.0f}%)')
    print('stop line types: ' + '  '.join(f'{n}={c}' for n, c in zip(names, n_sl) if c))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--map_path', default=os.environ.get('NUPLAN_MAPS_ROOT', 'nuplan/dataset/maps'))
    p.add_argument('--apply', action='store_true'); p.add_argument('--force', action='store_true')
    p.add_argument('--limit', type=int, default=0); p.add_argument('--shard', default='')
    main(p.parse_args())

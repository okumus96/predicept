import argparse, glob, os, time
import numpy as np
import torch
from tqdm import tqdm
from Predicept.channels_v3 import junction_crossing_conflict, A_CONFLICTS_WITH_PATH

SRC_VERSION = "channels-v3.0"
DST_VERSION = "channels-v3.2"
KEYS = ("neighbor_agents_past", "ego_agent_past", "c_lat_candidates", "intersections")


@torch.no_grad()
def branch2(datas, N):
    T = lambda k: torch.stack([torch.from_numpy(d[k]).float() for d in datas])
    return junction_crossing_conflict(T("neighbor_agents_past")[:, :N], T("ego_agent_past"),
                                      T("c_lat_candidates"), T("intersections")).numpy()


def main(a):
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    assert files, f"no npz found: {a.data}"
    if a.shard:
        k, n = map(int, a.shard.split("/")); files = files[k::n]
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} files (apply={a.apply}, {SRC_VERSION} -> {DST_VERSION})", flush=True)
    n_done = n_skip = n_err = 0; n_new = n_old = 0
    verified = False
    t0 = time.time()
    pbar = tqdm(total=len(files), unit="npz", mininterval=30.0)
    for i in range(0, len(files), a.batch_size):
        chunk = files[i:i + a.batch_size]; pbar.update(len(chunk))
        datas, keep = [], []
        for f in chunk:
            try:
                d = dict(np.load(f, allow_pickle=True))
            except Exception as e:
                print(f"[ERR-READ] {f}: {e}", flush=True); n_err += 1; continue
            v = str(d.get("channels_version_v3", ""))
            if v == DST_VERSION:
                n_skip += 1; continue
            assert v == SRC_VERSION, f"unexpected version '{v}': {f}"
            datas.append(d); keep.append(f)
        if not datas:
            continue
        b2 = branch2(datas, a.num_neighbors)
        for b, (f, d) in enumerate(zip(keep, datas)):
            old_keys = {k: d[k] for k in d if k not in ("channel_active_gt_v3", "channel_active_gf_v3",
                                                        "channels_version_v3")}
            gt = d["channel_active_gt_v3"].copy(); gf = d["channel_active_gf_v3"].copy()
            n_old += int(gt[:, A_CONFLICTS_WITH_PATH].sum())
            gt[:, A_CONFLICTS_WITH_PATH] |= b2[b]
            gf[:, A_CONFLICTS_WITH_PATH] |= b2[b]
            n_new += int(gt[:, A_CONFLICTS_WITH_PATH].sum())
            d["channel_active_gt_v3"] = gt; d["channel_active_gf_v3"] = gf
            d["channels_version_v3"] = DST_VERSION
            if a.apply:
                tmp = f + ".tmp.npz"
                try:
                    np.savez(tmp, **d); os.replace(tmp, f)
                    if not verified:
                        back = np.load(f, allow_pickle=True)
                        assert set(back.files) == set(d.keys()), f"KEY MISMATCH: {set(back.files) ^ set(d.keys())}"
                        for k, v_ in old_keys.items():
                            assert np.array_equal(np.asarray(back[k]), np.asarray(v_)), f"EXISTING KEY CHANGED: {k}"
                        print(f"[VERIFY] first file: {len(back.files)} keys, {len(old_keys)} untouched keys bit-identical",
                              flush=True)
                        verified = True
                except Exception as e:
                    if os.path.exists(tmp): os.remove(tmp)
                    print(f"[ERR-WRITE] {f}: {e}", flush=True); n_err += 1; continue
            n_done += 1
        if n_done and (i // max(1, a.batch_size)) % 50 == 0:
            el = (time.time() - t0) / 60
            print(f"[progress] {i+len(chunk)}/{len(files)}  {el:.1f} min  {n_done/max(el*60,1e-9):.1f} npz/s "
                  f"processed={n_done} skipped={n_skip} errors={n_err}", flush=True)
    pbar.close()
    print(f"DONE: processed={n_done} skipped={n_skip} errors={n_err}", flush=True)
    print(f"conflictsWithPath activations (GT): {n_old} -> {n_new}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=str, default="")
    p.add_argument("--apply", action="store_true")
    main(p.parse_args())

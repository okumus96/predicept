import argparse, glob, numpy as np
p = argparse.ArgumentParser(); p.add_argument("--pattern", required=True); p.add_argument("--out", required=True)
a = p.parse_args()
parts = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(a.pattern))]
assert parts, a.pattern
keys_cat = ['agent', 'col_refpath', 'col_gt', 'col_gt_ttc', 'map', 'files', 'raw', 'valid']
out = {k: np.concatenate([z[k] for z in parts]) for k in keys_cat}
order = np.argsort(out['files']); out = {k: v[order] for k, v in out.items()}
assert len(np.unique(out['files'])) == len(out['files']), "duplicate file names"
for k in ('ag_classes', 'mp_classes', 'raw_names'):
    for z in parts: assert list(z[k]) == list(parts[0][k])
    out[k] = parts[0][k]
np.savez(a.out, **out)
ag, mp = out['agent'], out['map']; agc, mpc = list(out['ag_classes']), list(out['mp_classes'])
print(f"[merged] {len(parts)} shard -> {len(out['files'])} scenes -> {a.out}")
for c in agc[1:]: print(f"   agent {c:22s} {int((ag == agc.index(c)).sum()):8d}")
for c in mpc[1:]: print(f"   map   {c:22s} {int((mp == mpc.index(c)).sum()):8d}")

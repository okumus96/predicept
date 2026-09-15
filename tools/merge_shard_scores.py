import argparse, glob, os, sys
import pandas as pd


def per_scenario(run_dir):
    f = glob.glob(os.path.join(run_dir, 'aggregator_metric', '*.parquet'))
    if not f:
        return None
    df = pd.read_parquet(f[0])
    return df[df['log_name'].notna()][['scenario', 'score']].copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pattern', help='glob matching the shard output folders')
    p.add_argument('--ref', default='', help='single-process reference run folder (per-scenario check)')
    p.add_argument('--expect', type=int, default=0, help='expected number of scenarios (e.g. 272)')
    a = p.parse_args()
    dirs = sorted(d for d in glob.glob(a.pattern) if os.path.isdir(d))
    assert dirs, f'no folder found: {a.pattern}'
    parts = []
    for d in dirs:
        t = per_scenario(d)
        print(f"  {os.path.basename(d):60s} {'-' if t is None else len(t)} scenarios")
        if t is not None:
            parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    dup = df['scenario'].duplicated().sum()
    print(f"\ntotal {len(df)} scenarios, duplicates {dup}, unique {df['scenario'].nunique()}")
    assert dup == 0, 'same scenario in two shards: sharding is broken'
    if a.expect:
        assert len(df) == a.expect, f'scenario count {len(df)} != expected {a.expect}: missing or extra shards'
    print(f"FINAL SCORE (mean over scenarios) = {df['score'].mean():.6f}")
    if a.ref:
        r = per_scenario(a.ref)
        assert r is not None, f'no aggregator output in the reference: {a.ref}'
        j = df.merge(r, on='scenario', suffixes=('_shard', '_ref'))
        d = (j['score_shard'] - j['score_ref']).abs()
        TOL = 1e-12
        print(f"\nscenarios shared with the reference {len(j)}: max |diff| {d.max():.2e}, identical {int((d == 0).sum())}/{len(j)}")
        print('RESULT:', 'OK: the sharded run matches the single-process run (differences within rounding)'
              if d.max() <= TOL else f'MISMATCH (> {TOL:g}), check')


if __name__ == '__main__':
    main()

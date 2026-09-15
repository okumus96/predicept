import argparse, glob, os, time
import numpy as np
import torch
from tqdm import tqdm
from Predicept.channels_v3 import (compute_agent_channels, compute_map_channels_v3,
                                    A_NAMES, M_NAMES, NUM_A, NUM_M)

VERSION = "channels-v3.2"


@torch.no_grad()
def gf_top1(gameformer, batch, device, num_neighbors):
    from train_planner import extract_neighbor_top1_futures
    inputs = {k: v.to(device) for k, v in batch.items()}
    top1, _, _ = extract_neighbor_top1_futures(gameformer, gameformer.encoder(inputs), num_neighbors)
    return top1.detach().cpu()


def v3_channels(datas, top1, N):
    T = lambda k: torch.stack([torch.from_numpy(d[k]).float() for d in datas])
    nbr, ego, ref = T("neighbor_agents_past")[:, :N], T("ego_agent_past"), T("c_lat_candidates")
    lanes, cw, routes = T("lanes"), T("crosswalks"), T("route_lanes")
    tl, intx, stops = T("lane_tl"), T("intersections"), T("stop_polygons")
    gt = T("neighbor_agents_future")[:, :N, :, :2]
    act_gt = compute_agent_channels(nbr, ego, ref, routes, cw, intx, stops, tl, lanes, neighbor_futures=gt)
    act_gf = compute_agent_channels(nbr, ego, ref, routes, cw, intx, stops, tl, lanes, neighbor_futures=top1)
    mact = compute_map_channels_v3(lanes, cw, routes, ref, tl, intx, stops,
                                   ego_v=ego[:, -1, 3:5].norm(dim=-1), ego_past=ego)
    return act_gt.bool(), act_gf.bool(), mact.bool()


def main(args):
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    assert files, f"no npz found: {args.data}"
    if args.shard:
        k, n = map(int, args.shard.split("/")); files = files[k::n]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} files (apply={args.apply}, force={args.force}, version={VERSION})", flush=True)

    from Predicept.predictor import GameFormer
    gf = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                    neighbors=args.num_neighbors)
    gf.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    gf.to(args.device).eval()

    N = args.num_neighbors
    n_done = n_skip = n_err = 0
    fire_gt = np.zeros(NUM_A, np.int64); fire_gf = np.zeros(NUM_A, np.int64); mfire = np.zeros(NUM_M, np.int64)
    verified_batch = False
    t0 = time.time()
    pbar = tqdm(total=len(files), unit="npz", desc="extract-v3", mininterval=30.0)
    for i in range(0, len(files), args.batch_size):
        chunk = files[i:i + args.batch_size]
        pbar.update(len(chunk))
        datas, keep = [], []
        for f in chunk:
            try:
                d = dict(np.load(f, allow_pickle=True))
            except Exception as e:
                print(f"[ERR-READ] {f}: {e}", flush=True); n_err += 1; continue
            if (not args.force) and str(d.get("channels_version_v3", "")) == VERSION:
                n_skip += 1; continue
            datas.append(d); keep.append(f)
        if not datas:
            continue
        T = lambda k: torch.stack([torch.from_numpy(d[k]).float() for d in datas])
        top1 = gf_top1(gf, {"ego_agent_past": T("ego_agent_past"), "neighbor_agents_past": T("neighbor_agents_past")[:, :N],
                            "map_lanes": T("lanes"), "map_crosswalks": T("crosswalks"), "route_lanes": T("route_lanes")},
                       args.device, N)
        act_gt, act_gf, mact = v3_channels(datas, top1, N)
        if not verified_batch:
            for b in range(min(len(datas), 4)):
                a1, g1, m1 = v3_channels([datas[b]], top1[b:b + 1], N)
                assert torch.equal(a1[0], act_gt[b]) and torch.equal(g1[0], act_gf[b]) and torch.equal(m1[0], mact[b]), \
                    f"batched != single-scene (file {keep[b]})"
            for b, d in enumerate(datas):
                if "channel_active_gt" in d:
                    v2col = d["channel_active_gt"][:N, 4].astype(bool)
                    assert np.array_equal(act_gt[b, :, 5].numpy() & v2col, v2col), \
                        f"v2 collision activation lost (branch 1 broken): {keep[b]}"
            print(f"[VERIFY] first batch: batched==single-scene ({min(len(datas),4)} scenes), collision(GT)==v2 column "
                  f"({sum('channel_active_gt' in d for d in datas)} scenes)", flush=True)
            verified_batch = True
        fire_gt += act_gt.sum((0, 1)).numpy(); fire_gf += act_gf.sum((0, 1)).numpy(); mfire += mact.sum((0, 1)).numpy()
        for b, (f, d) in enumerate(zip(keep, datas)):
            old_keys = set(d.keys())
            d["channel_active_gt_v3"] = act_gt[b].numpy()
            d["channel_active_gf_v3"] = act_gf[b].numpy()
            d["map_channel_active_v3"] = mact[b].numpy()
            d["channels_version_v3"] = VERSION
            if args.apply:
                tmp = f + ".tmp.npz"
                try:
                    np.savez(tmp, **d)
                    os.replace(tmp, f)
                    if n_done == 0:
                        back = np.load(f, allow_pickle=True)
                        assert set(back.files) == set(d.keys()), f"KEY MISMATCH: {set(back.files) ^ set(d.keys())}"
                        for k in old_keys:
                            assert np.array_equal(np.asarray(back[k]), np.asarray(d[k])), f"EXISTING KEY CHANGED: {k}"
                        print(f"[VERIFY] first file: {len(back.files)} keys, {len(old_keys)} existing keys bit-identical", flush=True)
                except Exception as e:
                    if os.path.exists(tmp): os.remove(tmp)
                    print(f"[ERR-WRITE] {f}: {e}", flush=True); n_err += 1; continue
            n_done += 1
        if (i // args.batch_size) % 200 == 0:
            el = time.time() - t0; rate = (i + len(chunk)) / max(el, 1e-6)
            print(f"[progress] {i + len(chunk)}/{len(files)}  {el/60:.1f} min  {rate:.1f} npz/s  ETA {(len(files) - i - len(chunk))/max(rate,1e-6)/60:.1f} min"
                  f"  processed={n_done} skipped={n_skip} errors={n_err}", flush=True)
    pbar.close()
    print(f"\nDONE: processed={n_done}  skipped(same version)={n_skip}  errors={n_err}  {'(DRY RUN -- nothing written)' if not args.apply else ''}", flush=True)
    print("ego->agent v3 fire (GT | GF):")
    for k in range(NUM_A): print(f"  {A_NAMES[k]:28s} {fire_gt[k]:8d} | {fire_gf[k]:8d}")
    print("ego->map v3 fire:")
    for k in range(NUM_M): print(f"  {M_NAMES[k]:28s} {mfire[k]:8d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--pretrained_path", default="training_log/normal/model_epoch_19_valADE_1.6487.pth")
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=str, default="", help="k/n: k-th shard of the file list")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", type=str, default="cuda:1")
    main(p.parse_args())

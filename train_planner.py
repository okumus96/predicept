import argparse
import csv
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Predicept.predictor import GameFormer
from Predicept.causal_graph import CausalPlanner
from Predicept.train_utils import DrivingData, imitation_loss, initLogging, set_seed


def extract_neighbor_top1_futures(gameformer, encoder_outputs, num_neighbors, return_ego=False):
    decoder_outputs, _ = gameformer.decoder(encoder_outputs)
    last_k = max(int(k.split('_')[1]) for k in decoder_outputs if 'interactions' in k)
    inter = decoder_outputs[f'level_{last_k}_interactions']        # [B, N+1, M, T, 4]
    scores = decoder_outputs[f'level_{last_k}_scores']             # [B, N+1, M]

    nbr_inter = inter[:, 1:1 + num_neighbors]                       # [B, N, M, T, 4]
    nbr_scores = scores[:, 1:1 + num_neighbors]                     # [B, N, M]
    best_mod = nbr_scores.argmax(-1)                                # [B, N]
    B, N, M, T, _ = nbr_inter.shape
    g = best_mod.view(B, N, 1, 1, 1).expand(-1, -1, 1, T, 2)
    top1_futures = torch.gather(nbr_inter[..., :2], 2, g).squeeze(2)  # [B, N, T, 2]

    current_states = encoder_outputs['actors'][:, 1:1 + num_neighbors, -1]  # [B, N, 5]
    nbr_valid = ~encoder_outputs['mask'][:, 1:1 + num_neighbors]            # [B, N]

    if return_ego:
        ego_best = scores[:, 0].argmax(-1)                                  # [B]
        g0 = ego_best.view(B, 1, 1, 1).expand(-1, 1, T, 2)
        ego_plan = torch.gather(inter[:, 0, ..., :2], 1, g0).squeeze(1)      # [B, T, 2]
        return top1_futures, current_states, nbr_valid, ego_plan
    return top1_futures, current_states, nbr_valid


def read_batch(batch, device):
    inputs = {
        "ego_agent_past": batch[0].to(device).float(),
        "neighbor_agents_past": batch[1].to(device).float(),
        "map_lanes": batch[2].to(device).float(),
        "map_crosswalks": batch[3].to(device).float(),
        "route_lanes": batch[4].to(device).float(),
    }
    ego_future = batch[5].to(device).float()
    neighbors_future = batch[6].to(device).float()
    c_lat_candidates = batch[7].to(device).float()
    if len(batch) > 12:
        inputs["channel_active"] = batch[9].to(device).bool()
        inputs["channel_evidence"] = batch[10].to(device).float()
        inputs["map_channel_active"] = batch[11].to(device).bool()
        inputs["map_channel_evidence"] = batch[12].to(device).float()
    if len(batch) > 14:
        inputs["decision_lon"] = batch[13].to(device).long()
        inputs["decision_lat"] = batch[14].to(device).long()
    if len(batch) > 16:
        inputs["l1_agent"] = batch[15].to(device).long()
        inputs["l1_map"] = batch[16].to(device).long()
    if len(batch) > 17:
        inputs["intersections"] = batch[17].to(device).float()
    return inputs, ego_future, neighbors_future, c_lat_candidates


def freeze_gameformer(gameformer):
    gameformer.eval()
    for parameter in gameformer.parameters():
        parameter.requires_grad = False


_MANEUVER = {'stationary': 0, 'straight': 1, 'turning_left': 2, 'turning_right': 3, 'U-turn_left': 4}
NUM_MANEUVERS = 5

from Predicept.decision_labels import (LON_CE_WEIGHT, LAT_CE_WEIGHT, LON_MERGE_MAP,
                                        LON_MERGED_CE_WEIGHT, NUM_LON, NUM_LAT, NUM_LON_MERGED,
                                        LON5_MAP, LAT5_MAP, LON5_CE_WEIGHT, LAT5_CE_WEIGHT,
                                        NUM_LON5, NUM_LAT5,
                                        LON4_MAP, LAT5V_MAP, LON4_CE_WEIGHT, LAT5V_CE_WEIGHT,
                                        NUM_LON4, NUM_LAT5V,
                                        LAT5L_MAP, LAT5L_CE_WEIGHT, NUM_LAT5L)
_LON_W = torch.tensor(LON_CE_WEIGHT, dtype=torch.float32)
_LAT_W = torch.tensor(LAT_CE_WEIGHT, dtype=torch.float32)
_LON_REMAP = None
_LAT_REMAP = None


def _resample_arc(xy, n):
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.repeat(xy[:1], n, axis=0)
    cum = cum / cum[-1]
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])], axis=1)


def _maneuver_one(xy, yaw):
    valid = ~np.all(xy == 0, axis=1)
    xy, yaw = xy[valid], yaw[valid]
    if len(xy) < 2:
        return _MANEUVER['stationary']
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if length < 3.0:
        return _MANEUVER['stationary']
    pts = _resample_arc(xy, int(length))
    tan = np.diff(pts, axis=0)
    tan = tan / np.clip(np.linalg.norm(tan, axis=1, keepdims=True), 1e-8, None)
    ang = np.arccos(np.clip((tan[:-1] * tan[1:]).sum(1), -1.0, 1.0))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    curv = ang / np.clip(seg[:-1], 1e-8, None)
    sign = np.sign(np.cross(tan[:-1], tan[1:]))
    i = int(np.argmax(curv))
    c = round(float(curv[i]), 2)
    s = float(sign[i])
    diff = round(float(abs(yaw[0] - yaw[-1])), 2)
    turning = (0.03 < c < 0.18 and diff > 0.2) or (0.1 < c < 0.18)
    uturn = c >= 0.18
    if turning or uturn:
        if s == 1.0:
            ctx = 'U-turn_left' if uturn else 'turning_left'
        elif s == -1.0:
            ctx = 'turning_right'
        else:
            ctx = 'straight'
    else:
        ctx = 'straight'
    return _MANEUVER[ctx]


def maneuver_labels(ego_future):
    ef = ego_future.detach().cpu().numpy()
    labs = [_maneuver_one(ef[b, :, :2], ef[b, :, 2]) for b in range(ef.shape[0])]
    return torch.tensor(labs, dtype=torch.long, device=ego_future.device)


L1_AG_WEIGHT = [0.27, 7.8, 40.0, 40.0, 39.0, 9.0]

L1_MASS_PRIORITY = [2, 3, 4, 5, 1]
L1_AG_WEIGHT_V3 = [0.26, 9.1, 10.0, 10.0, 10.0]
L1_MP_WEIGHT_V3 = [0.26, 8.8, 10.0, 10.0, 10.0]
L1_MP_WEIGHT_V3_MAP3 = [0.25, 10.0, 10.0]
L1_MASS_PRIORITY_V3 = [2, 3, 4, 1]
L1_MP_WEIGHT = None


def causal_loss_and_metrics(out, ego_future, lambda_kld, lambda_ci, lambda_mask, lambda_recon=0.0,
                            neighbors_future=None, lambda_nbr=0.0, lambda_budget=0.0, budget_rho=0.2, budget_lo=0.0,
                            lambda_bc=0.0, lambda_peak=0.0, peak_tau=0.5,
                            lambda_conflict=0.0, conflict_terms='all', dec_labels=None,
                            lambda_attr=0.0, l1_labels=None, lambda_mass=0.0, mass_floor=0.0):
    gt = ego_future[:, None]                                       # [B, 1, 80, 3]
    l_traj, _, best_mode = imitation_loss(out['traj'], out['score'], gt)
    dod_meta = out.get('psi_lon_cas') is not None

    def _uniform_ci(logits):
        lp = F.log_softmax(logits, dim=-1)
        u = torch.ones_like(logits) / logits.shape[-1]
        kl = F.kl_div(input=lp, target=u, reduction='batchmean', log_target=False)
        e = -(lp.exp() * lp).sum(-1).mean()
        return kl, e

    if dod_meta:
        lon_star, lat_star = dec_labels
        if _LON_REMAP is not None:
            lon_star = _LON_REMAP.to(lon_star.device)[lon_star]
        if _LAT_REMAP is not None:
            lat_star = _LAT_REMAP.to(lat_star.device)[lat_star]
        w_lon = _LON_W.to(out['psi_lon_cas'].device)
        w_lat = _LAT_W.to(out['psi_lat_cas'].device)
        l_kld = 0.5 * (F.cross_entropy(out['psi_lon_cas'], lon_star, weight=w_lon)
                       + F.cross_entropy(out['psi_lat_cas'], lat_star, weight=w_lat))
        kl1, e1 = _uniform_ci(out['psi_lon_cfd'])
        kl2, e2 = _uniform_ci(out['psi_lat_cfd'])
        l_kl, ent = 0.5 * (kl1 + kl2), 0.5 * (e1 + e2)
        l_ci = l_kl + 0.1 * (-ent)
        m_star = None
    else:
        m_star = maneuver_labels(ego_future)

        l_kld = F.cross_entropy(out['psi_cas'], m_star)

        l_kl, ent = _uniform_ci(out['psi_cfd'])
        l_ci = l_kl + 0.1 * (-ent)

    def _soft_mask(M_cas, M_cfd, valid):
        vs = valid
        comp = F.mse_loss((M_cas + M_cfd)[vs], torch.ones_like(M_cas[vs]))          # -> 1
        excl = F.mse_loss((M_cas * M_cfd)[vs], torch.zeros_like(M_cas[vs]))         # -> 0
        rows = vs.any(dim=-1)
        cs, fs = M_cas.sum(-1)[rows], M_cfd.sum(-1)[rows]
        ones = torch.ones_like(cs)
        norm = F.mse_loss(cs, ones) + F.mse_loss(fs, ones)
        return comp, excl, norm

    comp_ag, excl_ag, norm_ag = _soft_mask(out['M_cas'], out['M_cfd'], out['nbr_valid'])
    comp_mp, excl_mp, norm_mp = _soft_mask(out['M_cas_map'], out['M_cfd_map'], out['map_valid'])
    l_comp, l_excl, l_norm = comp_ag + comp_mp, excl_ag + excl_mp, norm_ag + norm_mp
    l_mask = l_comp + l_excl

    nvb_b = out['nbr_valid']
    g_mean = (out['M_cas'] * nvb_b).sum() / nvb_b.sum().clamp(min=1)
    l_budget = F.relu(g_mean - budget_rho) + F.relu(budget_lo - g_mean)

    l_recon = F.mse_loss(out['recon_pred'], out['f_all'].detach())

    l_nbr = torch.zeros((), device=out['traj'].device)
    if neighbors_future is not None:
        N = out['nbr_pred'].shape[1]
        gt_nbr = neighbors_future[:, :N, :, :2]                              # [B,N,T,2]
        vmask = torch.ne(gt_nbr, 0).any(-1) & out['nbr_valid'][:, :N, None]  # [B,N,T]
        per = F.smooth_l1_loss(out['nbr_pred'], gt_nbr, reduction='none').mean(-1)   # [B,N,T]
        l_nbr = (per * vmask).sum() / vmask.sum().clamp(min=1)

    eps = 1e-12
    def _bc(M_cas, M_cfd, valid):
        vf = valid.float()
        return (((M_cas.clamp(min=eps) * M_cfd.clamp(min=eps)).sqrt() * vf).sum(-1)).mean()
    l_bc = _bc(out['M_cas'], out['M_cfd'], out['nbr_valid']) \
         + _bc(out['M_cas_map'], out['M_cfd_map'], out['map_valid'])

    l_peak = F.relu(out['M_cas_ent'] - peak_tau).mean() + F.relu(out['M_cas_map_ent'] - peak_tau).mean()

    l_conflict = torch.zeros((), device=out['traj'].device)
    if lambda_conflict > 0.0 and out.get('conflict') is not None:
        pen_idx = [2] if conflict_terms == 'reach' else [0, 1, 2]
        penalty = out['conflict'][..., pen_idx].mean(-1)
        l_conflict = (out['M_cas'] * penalty).sum(-1).mean()

    nv_f = out['nbr_valid'].float()
    l_attr = torch.zeros((), device=out['traj'].device)
    if lambda_attr > 0 and l1_labels is not None and out.get('l1_ag') is not None:
        y_ag, y_mp = l1_labels
        va, vm = out['gated_valid'], out['gated_map_valid']
        la, lm = out['l1_ag'], out['l1_mp']
        w_ag = torch.tensor(L1_AG_WEIGHT, device=la.device, dtype=la.dtype)
        if va.any():
            l_attr = l_attr + F.cross_entropy(la[va], y_ag[:, :la.shape[1]][va], weight=w_ag)
        if vm.any():
            w_mp = (torch.tensor(L1_MP_WEIGHT, device=lm.device, dtype=lm.dtype) if L1_MP_WEIGHT is not None else None)
            l_attr = l_attr + F.cross_entropy(lm[vm], y_mp[:, :lm.shape[1]][vm], weight=w_mp)

    l_mass = torch.zeros((), device=out['traj'].device)
    if lambda_mass > 0 and l1_labels is not None and out.get('M_cas') is not None:
        y_ag, y_mp = l1_labels
        terms = []
        raw_ag = out.get('M_cas_raw'); raw_mp = out.get('M_cas_map_raw')
        joint = raw_ag is not None
        for Mc, Mraw, va, y in ((out['M_cas'], raw_ag, out['gated_valid'], y_ag),
                                (out['M_cas_map'], raw_mp, out['gated_map_valid'], y_mp)):
            if Mc is None:
                continue
            y = y[:, :Mc.shape[1]]
            tgt = (y > 0) & va
            has = tgt.any(-1)
            if not bool(has.any()):
                continue
            if joint:
                share = (Mraw * tgt.float()).sum(-1)                   # [B]
                terms.append(-torch.log(share[has].clamp(min=1e-6)).mean())
            else:
                q = tgt.float(); q = q / q.sum(-1, keepdim=True).clamp(min=1.0)
                p = Mc * va.float(); p = p / p.sum(-1, keepdim=True).clamp(min=1e-6)
                terms.append(-(q[has] * torch.log(p[has] + 1e-8)).sum(-1).mean())
        if terms:
            l_mass = torch.stack(terms).mean()
    loss = (l_traj + lambda_kld * l_kld + lambda_ci * l_ci + lambda_mask * l_mask
            + lambda_attr * l_attr + lambda_mass * l_mass
            + lambda_recon * l_recon + lambda_nbr * l_nbr + lambda_budget * l_budget + lambda_bc * l_bc + lambda_peak * l_peak
            + lambda_conflict * l_conflict)

    with torch.no_grad():
        traj_xy = out['traj'][:, 0, :, :, :2]                     # [B, M, 80, 2]
        gt_xy = gt[:, 0, None, :, :2]                             # [B, 1, 80, 2]
        ade = torch.norm(traj_xy - gt_xy, dim=-1).mean(-1)        # [B, M]
        best = ade.argmin(-1)                                     # [B]
        minade = ade.gather(1, best[:, None]).mean().item()
        fde = torch.norm(traj_xy[:, :, -1] - gt_xy[:, :, -1], dim=-1)  # [B, M]
        minfde = fde.gather(1, best[:, None]).mean().item()

        if dod_meta:
            cas_acc = 0.5 * ((out['psi_lon_cas'].argmax(-1) == lon_star).float().mean()
                             + (out['psi_lat_cas'].argmax(-1) == lat_star).float().mean()).item()
            cfd_acc = 0.5 * ((out['psi_lon_cfd'].argmax(-1) == lon_star).float().mean()
                             + (out['psi_lat_cfd'].argmax(-1) == lat_star).float().mean()).item()
        else:
            cas_acc = (out['psi_cas'].argmax(-1) == m_star).float().mean().item()
            cfd_acc = (out['psi_cfd'].argmax(-1) == m_star).float().mean().item()
        nvb = out['nbr_valid']
        mcas_peak = out['M_cas'].masked_fill(~nvb, 0.0).max(-1).values.mean().item()
        mcas_map_peak = out['M_cas_map'].masked_fill(~out['map_valid'], 0.0).max(-1).values.mean().item()
        q_bar = ((out['M_cas'] * nv_f).sum() / nv_f.sum().clamp(min=1.0)).item()
        mcfd_peak = out['M_cfd'].masked_fill(~nvb, 0.0).max(-1).values.mean().item()
        fcfd_var = out['f_cfd'].var(dim=0).mean().item()
        fcas_var = out['f_cas'].var(dim=0).mean().item()
        n_valid = nvb.sum(-1).clamp(min=1).float()                                     # [B]
        unif = (1.0 / n_valid).mean().item()

        l1_acc = l1_rec = l1_macc = 0.0
        if out.get('l1_ag') is not None and dec_labels is not None and l1_labels is not None:
            ya, ym = l1_labels
            if ya is not None:
                va2, vm2 = out['gated_valid'], out['gated_map_valid']
                if va2.any():
                    pa = out['l1_ag'].argmax(-1)[va2]; ta = ya[:, :out['l1_ag'].shape[1]][va2]
                    l1_acc = (pa == ta).float().mean().item()
                    nz = ta > 0
                    l1_rec = (pa[nz] == ta[nz]).float().mean().item() if nz.any() else 0.0
                if vm2.any():
                    pm = out['l1_mp'].argmax(-1)[vm2]; tm = ym[:, :out['l1_mp'].shape[1]][vm2]
                    l1_macc = (pm == tm).float().mean().item()
        mcas_ent = out['M_cas_ent'].mean().item()
        mcas_headent = out['M_cas_headent'].mean().item()
        mcfd_ent = out['M_cfd_ent'].mean().item()
        mcfd_headent = out['M_cfd_headent'].mean().item()
        mcas_map_ent = out['M_cas_map_ent'].mean().item()
        mcas_map_headent = out['M_cas_map_headent'].mean().item()
        mcfd_map_ent = out['M_cfd_map_ent'].mean().item()
        mcfd_map_headent = out['M_cfd_map_headent'].mean().item()

        gate_cos_per_layer = out['gate_cos'].mean(0)
        gate_cos_last = gate_cos_per_layer[-1].item()

        if dod_meta:
            lp1 = F.log_softmax(out['psi_lon_cas'], dim=-1)
            lp2 = F.log_softmax(out['psi_lat_cas'], dim=-1)
            casent = 0.5 * (-(lp1.exp() * lp1).sum(-1).mean()
                            - (lp2.exp() * lp2).sum(-1).mean()).item()
        else:
            log_p_cas = F.log_softmax(out['psi_cas'], dim=-1)
            casent = -(log_p_cas.exp() * log_p_cas).sum(-1).mean().item()
        entgap = ent.item() - casent

    metrics = {
        'loss': loss.item(), 'traj': l_traj.item(), 'kld': l_kld.item(),
        'kl': l_kl.item(), 'ent': ent.item(), 'casent': casent, 'entgap': entgap,
        'ci': l_ci.item(), 'mask': l_mask.item(),
        'comp': l_comp.item(), 'excl': l_excl.item(), 'norm': l_norm.item(),
        'recon': l_recon.item(), 'nbr': l_nbr.item(), 'budget': l_budget.item(),
        'bc': l_bc.item(), 'peak_hinge': l_peak.item(), 'conflict': l_conflict.item(),
        'agmass': (out['agent_mass'].mean().item() if out.get('agent_mass') is not None else 0.0),
        'egomass': (out['ego_mass'].mean().item() if out.get('ego_mass') is not None else 0.0),
        'gcas_mean': g_mean.item(),
        'gcfd_mean': ((out['M_cfd'] * nvb_b).sum() / nvb_b.sum().clamp(min=1)).item(),
        'gcas_frac05': (((out['M_cas'] > 0.5) & nvb_b).sum() / nvb_b.sum().clamp(min=1)).item(),
        'minADE': minade, 'minFDE': minfde, 'casacc': cas_acc, 'cfdacc': cfd_acc,
        'attr': l_attr.item(), 'mass': l_mass.item(), 'l1acc': l1_acc, 'l1rec': l1_rec, 'l1macc': l1_macc,
        'mcas_peak': mcas_peak, 'mcas_map_peak': mcas_map_peak, 'qbar': q_bar,
        'mcfd_peak': mcfd_peak, 'unif': unif,
        'fcfd_var': fcfd_var, 'fcas_var': fcas_var,
        'mcas_ent': mcas_ent, 'mcas_headent': mcas_headent,
        'mcfd_ent': mcfd_ent, 'mcfd_headent': mcfd_headent,
        'mcas_map_ent': mcas_map_ent, 'mcas_map_headent': mcas_map_headent,
        'mcfd_map_ent': mcfd_map_ent, 'mcfd_map_headent': mcfd_map_headent,
        'gate_cos_last': gate_cos_last,
    }
    for i, v in enumerate(gate_cos_per_layer.tolist()):
        metrics[f'gate_cos_l{i}'] = v
    return loss, metrics


def _run_epoch(data_loader, gameformer, causal, device, num_neighbors,
               lambda_kld, lambda_ci, lambda_mask, lambda_recon=0.0, lambda_nbr=0.0,
               lambda_budget=0.0, budget_rho=0.2, budget_lo=0.0,
               lambda_bc=0.0, lambda_peak=0.0, peak_tau=0.5, lambda_conflict=0.0, conflict_terms='all',
               ego_corridor='refpath', optimizer=None, desc="Training", lambda_attr=0.0,
               lambda_mass=0.0, mass_floor=0.25):
    train = optimizer is not None
    causal.train() if train else causal.eval()
    gameformer.eval()
    agg = defaultdict(list)

    with tqdm(data_loader, desc=desc, unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, neighbors_future, ref_path = read_batch(batch, device)

            with torch.no_grad():
                encoder_outputs = gameformer.encoder(inputs)
                if ego_corridor == 'gf':
                    top1_fut, nbr_states, _, ego_plan = extract_neighbor_top1_futures(
                        gameformer, encoder_outputs, num_neighbors=num_neighbors, return_ego=True)
                    ref_path = ego_plan[:, None]                             # [B,1,T,2]
                else:
                    top1_fut, nbr_states, _ = extract_neighbor_top1_futures(
                        gameformer, encoder_outputs, num_neighbors=num_neighbors
                    )

            with torch.set_grad_enabled(train):
                out = causal(encoder_outputs, inputs, num_agents=num_neighbors + 1,
                             neighbor_futures=top1_fut, neighbor_states=nbr_states, ref_path=ref_path)
                loss, metrics = causal_loss_and_metrics(out, ego_future, lambda_kld,
                                                         lambda_ci, lambda_mask, lambda_recon,
                                                         neighbors_future, lambda_nbr,
                                                         lambda_budget, budget_rho, budget_lo,
                                                         lambda_bc, lambda_peak, peak_tau,
                                                         lambda_conflict, conflict_terms,
                                                         dec_labels=(inputs.get('decision_lon'),
                                                                     inputs.get('decision_lat')),
                                                         lambda_attr=lambda_attr,
                                                         lambda_mass=lambda_mass, mass_floor=mass_floor,
                                                         l1_labels=((inputs['l1_agent'], inputs['l1_map'])
                                                                    if 'l1_agent' in inputs else None))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(causal.parameters(), 5.0)
                optimizer.step()

            for key, val in metrics.items():
                agg[key].append(val)
            data_epoch.set_postfix(
                loss=f"{np.mean(agg['loss']):.3f}", minADE=f"{np.mean(agg['minADE']):.3f}",
                casacc=f"{np.mean(agg['casacc']):.3f}", cfdacc=f"{np.mean(agg['cfdacc']):.3f}",
                peak=f"{np.mean(agg['mcas_peak']):.3f}", cfdpk=f"{np.mean(agg['mcfd_peak']):.3f}",
                l1rec=f"{np.mean(agg['l1rec']):.3f}",
                entgap=f"{np.mean(agg['entgap']):.3f}",
                unif=f"{np.mean(agg['unif']):.3f}",
            )

    return {key: float(np.mean(val)) for key, val in agg.items()}


def model_training(args):
    global _LON_W, _LON_REMAP, _LAT_W, _LAT_REMAP
    if args.dod_meta and args.lon_merge:
        _LON_W = torch.tensor(LON_MERGED_CE_WEIGHT, dtype=torch.float32)
        _LON_REMAP = torch.tensor(LON_MERGE_MAP, dtype=torch.long)
    if args.dec_moe:
        assert args.dod_meta and not args.lon_merge, "dec_moe requires dod_meta=1, lon_merge=0"
        _LON_W = torch.tensor(LON5_CE_WEIGHT, dtype=torch.float32)
        _LON_REMAP = torch.tensor(LON5_MAP, dtype=torch.long)
        _LAT_W = torch.tensor(LAT5_CE_WEIGHT, dtype=torch.float32)
        _LAT_REMAP = torch.tensor(LAT5_MAP, dtype=torch.long)
    if args.dod_tf:
        assert args.dod_meta and not args.lon_merge and not args.dec_moe, \
            "dod_tf requires dod_meta=1, lon_merge=0, dec_moe=0"
    if args.lat_moe:
        assert args.dod_meta and not (args.lon_merge or args.dec_moe or args.dod_tf), \
            "lat_moe requires dod_meta=1 and cannot be combined with lon_merge/dec_moe/dod_tf"
        _LON_W = torch.tensor(LON4_CE_WEIGHT, dtype=torch.float32)
        _LON_REMAP = torch.tensor(LON4_MAP, dtype=torch.long)
        if int(args.lat_moe) >= 2:
            _LAT_W = torch.tensor(LAT5L_CE_WEIGHT, dtype=torch.float32)
            _LAT_REMAP = torch.tensor(LAT5L_MAP, dtype=torch.long)
        else:
            _LAT_W = torch.tensor(LAT5V_CE_WEIGHT, dtype=torch.float32)
            _LAT_REMAP = torch.tensor(LAT5V_MAP, dtype=torch.long)

    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path + "train.log")

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))
    logging.info("Config: {}".format({k: v for k, v in sorted(vars(args).items())}))

    set_seed(args.seed)

    global L1_AG_WEIGHT, L1_MASS_PRIORITY, L1_MP_WEIGHT
    if args.channel_set == 'v3':
        if args.num_l1_ag == 6: args.num_l1_ag = 5
        if args.num_l1_mp == 2: args.num_l1_mp = 3
        assert args.num_l1_ag == 5, 'the v3 L1 agent vocabulary has 5 classes'
        assert args.num_l1_mp in (3, 5), 'the v3 L1 map vocabulary has 3 or 5 classes'
        L1_AG_WEIGHT, L1_MASS_PRIORITY = L1_AG_WEIGHT_V3, L1_MASS_PRIORITY_V3
        L1_MP_WEIGHT = L1_MP_WEIGHT_V3_MAP3 if args.num_l1_mp == 3 else L1_MP_WEIGHT_V3
        logging.info('channel_set=v3: num_l1_ag=%d num_l1_mp=%d, L1 weights %s / %s, mass priority %s',
                     args.num_l1_ag, args.num_l1_mp, L1_AG_WEIGHT, L1_MP_WEIGHT, L1_MASS_PRIORITY)

    gameformer = GameFormer(
        encoder_layers=args.encoder_layers,
        decoder_levels=args.decoder_levels,
        neighbors=args.num_neighbors,
    )
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    gameformer = gameformer.to(args.device)
    freeze_gameformer(gameformer)

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, dropout=args.dropout,
                           recon_drop=(args.recon_drop if args.lambda_recon > 0 else 0.0),
                           num_neighbors=args.num_neighbors, gate=args.gate,
                           conflict_feats=args.conflict_feats, conflict_bias=args.conflict_bias,
                           compute_conflict=(args.compute_conflict or args.lambda_conflict > 0),
                           aligned_mode=args.aligned_mode, ego_residual=args.ego_residual,
                           joint_softmax=args.joint_softmax,
                           nbr_enrich=args.nbr_enrich,
                           gate_channels=args.gate_channels, typed_kv=args.typed_kv,
                           channel_evidence=args.channel_evidence,
                           gate_trust=args.gate_trust,
                           l1_drop_input=args.l1_drop_input,
                           dod_meta=args.dod_meta, dec_moe=args.dec_moe, dod_tf=args.dod_tf,
                           lat_moe=args.lat_moe,
                           l1=args.l1, l1_bottleneck=args.l1_bottleneck,
                           num_l1_ag=args.num_l1_ag, num_l1_mp=args.num_l1_mp,
                           channel_set=args.channel_set, psi_l0=args.psi_l0, psi_ego=args.psi_ego,
                           ego_family=args.ego_family, ego_route=args.ego_route, psi_linear=args.psi_linear,
                           num_lon=(NUM_LON4 if args.lat_moe else NUM_LON5 if args.dec_moe
                                    else NUM_LON_MERGED if args.lon_merge else NUM_LON),
                           num_lat=(NUM_LAT5V if args.lat_moe
                                    else NUM_LAT5 if args.dec_moe else NUM_LAT)).to(args.device)

    optimizer = optim.AdamW(causal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)

    train_set = DrivingData(args.train_set + "/*.npz", args.num_neighbors,
                            l1_labels=(args.l1_labels or None), channel_set=args.channel_set)
    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors,
                            l1_labels=(args.l1_valid_labels or args.l1_labels or None),
                            channel_set=args.channel_set)

    if args.dup_boost > 0:
        cache = np.load(args.label_cache)
        lon_raw = torch.from_numpy(cache['lon'].astype(np.int64))
        lat_raw = torch.from_numpy(cache['lat'].astype(np.int64))
        assert len(lon_raw) == len(train_set), \
            f"label cache {len(lon_raw)} != train_set {len(train_set)}; rebuild the cache"
        boost = ((lat_raw == 2) | (lat_raw == 3)
                 | ((lon_raw >= 1) & (lon_raw <= 4)))
        idx = torch.arange(len(train_set))
        dup = idx[boost].repeat(args.dup_boost - 1)
        idx = torch.cat([idx, dup])
        logging.info(f"dup_boost x{args.dup_boost}: {int(boost.sum())} scenes duplicated "
                     f"({100.0*float(boost.float().mean()):.1f}%), epoch {len(train_set)} -> {len(idx)}")
        train_set = torch.utils.data.Subset(train_set, idx.tolist())
        def _eff_w(raw, remap, k):
            lab = remap[raw] if remap is not None else raw
            lab = torch.cat([lab, (remap[raw] if remap is not None else raw)[boost].repeat(
                args.dup_boost - 1)])
            cnt = torch.bincount(lab, minlength=k).float()
            w = len(lab) / (k * cnt.clamp(min=1.0))
            w = torch.where(cnt > 0, w.clamp(max=10.0), torch.ones_like(w))
            return w.float()
        _LON_W = _eff_w(lon_raw, _LON_REMAP, len(_LON_W))
        _LAT_W = _eff_w(lat_raw, _LAT_REMAP, len(_LAT_W))
        logging.info(f"effective CE weights: lon={[round(float(x),2) for x in _LON_W]} "
                     f"lat={[round(float(x),2) for x in _LAT_W]}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))

    for epoch in range(args.train_epochs):
        logging.info(f"Epoch {epoch + 1}/{args.train_epochs}")
        causal.ss_p = args.ss_max * epoch / max(1, args.train_epochs - 1)
        train_m = _run_epoch(train_loader, gameformer, causal, args.device, args.num_neighbors,
                             args.lambda_kld, args.lambda_ci, args.lambda_mask, args.lambda_recon, args.lambda_nbr,
                             args.lambda_budget, args.budget_rho, args.budget_lo,
                             args.lambda_bc, args.lambda_peak, args.peak_tau, args.lambda_conflict, args.conflict_terms,
                             args.ego_corridor, optimizer=optimizer, desc="Training",
                             lambda_attr=args.lambda_attr, lambda_mass=args.lambda_mass, mass_floor=args.mass_floor)
        val_m = _run_epoch(valid_loader, gameformer, causal, args.device, args.num_neighbors,
                           args.lambda_kld, args.lambda_ci, args.lambda_mask, args.lambda_recon, args.lambda_nbr,
                           args.lambda_budget, args.budget_rho, args.budget_lo,
                           args.lambda_bc, args.lambda_peak, args.peak_tau, args.lambda_conflict, args.conflict_terms,
                           args.ego_corridor, optimizer=None, desc="Validation",
                           lambda_attr=args.lambda_attr, lambda_mass=args.lambda_mass, mass_floor=args.mass_floor)

        log = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]}
        log.update({f"train-{k}": v for k, v in train_m.items()})
        log.update({f"val-{k}": v for k, v in val_m.items()})

        log_file = f"./training_log/{args.name}/train_log.csv"
        write_header = epoch == 0
        with open(log_file, "w" if write_header else "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(log.keys())
            writer.writerow(log.values())

        logging.info(
            f"train: minADE={train_m['minADE']:.3f} casacc={train_m['casacc']:.3f} "
            f"cfdacc={train_m['cfdacc']:.3f} l1acc={train_m['l1acc']:.3f} l1rec={train_m['l1rec']:.3f} "
            f"l1macc={train_m['l1macc']:.3f} peak={train_m['mcas_peak']:.3f} cfdpk={train_m['mcfd_peak']:.3f} "
            f"hgap={train_m['mcas_ent'] - train_m['mcas_headent']:.3f} "
            f"hgap_mp={train_m['mcas_map_ent'] - train_m['mcas_map_headent']:.3f} "
            f"gcos={train_m['gate_cos_last']:.3f} "
            f"{('agmass=%.3f ' % train_m['agmass']) if train_m.get('agmass') else ''}| "
            f"val: minADE={val_m['minADE']:.3f} casacc={val_m['casacc']:.3f} "
            f"cfdacc={val_m['cfdacc']:.3f} peak={val_m['mcas_peak']:.3f} cfdpk={val_m['mcfd_peak']:.3f} "
            f"hgap={val_m['mcas_ent'] - val_m['mcas_headent']:.3f} "
            f"hgap_mp={val_m['mcas_map_ent'] - val_m['mcas_map_headent']:.3f} "
            f"gcos={val_m['gate_cos_last']:.3f} "
            f"{('agmass=%.3f ' % val_m['agmass']) if val_m.get('agmass') else ''}"
            f"cfdvar={val_m['fcfd_var'] / max(val_m['fcas_var'], 1e-8):.3f}"
        )

        scheduler.step()

        torch.save(
            causal.state_dict(),
            f"training_log/{args.name}/causal_epoch_{epoch + 1}_minADE_{val_m['minADE']:.4f}.pth",
        )
        logging.info(f"CausalPlanner saved in training_log/{args.name}\n")


# Predicept training configuration
CONFIG = dict(
    seed=3407, train_epochs=20, batch_size=32, learning_rate=1e-4, weight_decay=0.01, dropout=0.1,
    encoder_layers=3, decoder_levels=2, num_neighbors=10,
    graph_layers=1, nbr_enrich=2, modes=6, ego_residual=0, gate='softmax',
    channel_set='v3', gate_channels=1, typed_kv=1, channel_evidence=0, gate_trust='all', joint_softmax=1,
    ego_family=1, ego_route=1, psi_l0=1, psi_ego=0, psi_linear=0,
    l1=1, l1_bottleneck=1, l1_drop_input=0, num_l1_ag=5, num_l1_mp=3,
    dod_meta=1, lat_moe=1, lon_merge=0, dec_moe=0, dod_tf=0, ss_max=0.5,
    lambda_kld=1.0, lambda_ci=0.5, lambda_mask=0.5, lambda_nbr=0.1, lambda_attr=0.5, lambda_mass=0.5, mass_floor=0.25,
    lambda_recon=0.0, recon_drop=0.5, lambda_bc=0.0, lambda_peak=0.0, peak_tau=0.5,
    lambda_budget=0.0, budget_rho=0.2, budget_lo=0.0,
    lambda_conflict=0.0, conflict_terms='all', conflict_feats=0, conflict_bias=0, compute_conflict=0,
    aligned_mode='straight', ego_corridor='refpath', dup_boost=0, label_cache='',
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Predicept")
    parser.add_argument("--name", type=str, default="predicept", help="run name (output folder under training_log/)")
    parser.add_argument("--train_set", type=str, required=True, help="processed training data folder")
    parser.add_argument("--valid_set", type=str, required=True, help="processed validation data folder")
    parser.add_argument("--pretrained_path", type=str, required=True, help="frozen GameFormer backbone checkpoint")
    parser.add_argument("--l1_labels", type=str, required=True, help="L1 concept labels of the training set")
    parser.add_argument("--l1_valid_labels", type=str, required=True, help="L1 concept labels of the validation set")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    for key, value in CONFIG.items():
        setattr(args, key, value)

    model_training(args)

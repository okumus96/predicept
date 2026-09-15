import numpy as np
import torch

from .plannerv2 import Planner as PlannerV2
from .planner_utils import *
from .observation import observation_adapter
from Predicept.predictor import GameFormer
from Predicept.causal_graph import CausalPlanner
from Predicept.decision_labels import decision_labels, LON4_MAP, LAT5V_MAP
from train_planner import extract_neighbor_top1_futures

from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class CausalRefinerPlanner(PlannerV2):
    """Predicept planner: frozen GameFormer backbone, Predicept decision and trajectory head,
    and the GameFormer-Planner lattice refiner."""

    def __init__(self, backbone_path, causal_path, psi_prior_alpha=0.0, device=None):
        super().__init__(model_path=causal_path, device=device, debug=False,
                         debug_dir=None, debug_max_plots=0, oracle_mode=False)
        self._backbone_path = backbone_path
        self._causal_path = causal_path
        self._num_neighbors = 10
        self._psi_prior_alpha = float(psi_prior_alpha)

    def name(self) -> str:
        return 'Predicept'

    def _initialize_model(self):
        self.backbone = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=self._num_neighbors)
        self.backbone.load_state_dict(torch.load(self._backbone_path, map_location=self._device))
        self.backbone.to(self._device).eval()
        self.causal = CausalPlanner(layers=1, modes=6, nbr_enrich=2, ego_residual=0,
                                    gate_channels=1, typed_kv=1, channel_evidence=0, gate_trust='all',
                                    dod_meta=1, dec_moe=0, lat_moe=1,
                                    l1=1, l1_bottleneck=1, l1_drop_input=0,
                                    num_l1_ag=5, num_l1_mp=3,
                                    channel_set='v3', ego_family=1, ego_route=1, psi_l0=1,
                                    num_lon=4, num_lat=5, uniform_mask=0, joint_softmax=1)
        missing, unexpected = self.causal.load_state_dict(
            torch.load(self._causal_path, map_location=self._device), strict=False)
        if missing or unexpected:
            print(f'[load] missing={list(missing)}  unexpected={list(unexpected)}')
        self.causal.psi_prior_alpha = self._psi_prior_alpha
        self.causal.to(self._device).eval()
        self.relevance_graph = None

    def _channels_ref_path(self, ego_state, traffic_light_data):
        c_lat, _ = self.get_multimodal_reference_paths2(
            ego_state, traffic_light_data, points_per_route=MAX_LEN * 10)
        if np.abs(c_lat).sum() < 1e-6:
            P = MAX_LEN * 10
            synth = np.zeros((5, P, 6), dtype=np.float32)
            synth[0, :, 0] = np.linspace(-2.0, float(MAX_LEN), P)
            synth[0, :, 4] = 15.0
            return torch.tensor(synth, dtype=torch.float32, device=self._device).unsqueeze(0)
        return torch.tensor(c_lat, dtype=torch.float32, device=self._device).unsqueeze(0)

    @torch.no_grad()
    def _run_causal(self, features, ch_ref_path):
        enc = self.backbone.encoder(features)
        top1, nbr_states, _ = extract_neighbor_top1_futures(self.backbone, enc, self._num_neighbors)
        return self.causal(enc, features, num_agents=self._num_neighbors + 1,
                           neighbor_futures=top1, neighbor_states=nbr_states,
                           also_cfd_plan=False, ref_path=ch_ref_path)

    @torch.no_grad()
    def _cc_pick(self, traj, score, out, ch_ref, fallback):
        M = traj.shape[0]
        xy = traj[:, :, :2].detach().cpu()
        d = xy[:, 1:] - xy[:, :-1]
        hd = torch.atan2(d[..., 1], d[..., 0])
        hd = torch.cat([hd[:, :1], hd], dim=1)
        plans = torch.cat([xy, hd.unsqueeze(-1)], dim=-1)
        rl, rt = decision_labels(plans, ch_ref.cpu().expand(M, -1, -1, -1))
        rl, rt = torch.tensor(LON4_MAP)[rl], torch.tensor(LAT5V_MAP)[rt]
        bl = int(out['psi_lon_cas'][0].argmax())
        bt = int(out['psi_lat_cas'][0].argmax())
        sc = score.detach().cpu()
        for ok in ((rl == bl) & (rt == bt), (rt == bt), (rl == bl)):
            if bool(ok.any()):
                s2 = sc.clone()
                s2[~ok] = -1e9
                return int(s2.argmax())
        return fallback

    def _causal_neural_plan(self, features, ch_ref_path):
        out = self._run_causal(features, ch_ref_path)
        traj = out['traj'][0, 0]
        best = int(out['score'][0, 0].argmax().item())
        if ch_ref_path is not None and out.get('psi_lon_cas') is not None:
            best = self._cc_pick(traj, out['score'][0, 0], out, ch_ref_path, best)
        xy = traj[best, :, :2].detach().cpu().numpy()
        diffs = np.diff(xy, axis=0)
        heading = np.arctan2(diffs[:, 1], diffs[:, 0])
        heading = np.concatenate([heading[:1], heading])
        plan = np.concatenate([xy, heading[:, None]], axis=1).astype(np.float32)
        return torch.from_numpy(plan).unsqueeze(0).to(self._device)

    def _plan(self, ego_state, history, traffic_light_data, observation):
        features = observation_adapter(history, traffic_light_data, self._map_api,
                                       self._route_roadblock_ids, self._device)
        ch_ref = self._channels_ref_path(ego_state, traffic_light_data)
        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        with torch.no_grad():
            _, _, predictions, scores, ego_cur, nbr_cur = self._get_prediction(features)
            plan = self._causal_neural_plan(features, ch_ref)

        if ref_path is None:
            plan_np = plan[0].detach().cpu().numpy()
            states = transform_predictions_to_states(plan_np, history.ego_states, self._future_horizon, DT)
            return InterpolatedTrajectory(states)

        final_plan = self._trajectory_planner.plan(ego_state, ego_cur, nbr_cur,
                                                   predictions, plan, scores, ref_path, observation)
        states = transform_predictions_to_states(final_plan, history.ego_states, self._future_horizon, DT)
        return InterpolatedTrajectory(states)

    def compute_planner_trajectory(self, current_input):
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        return self._plan(ego_state, history, traffic_light_data, observation)

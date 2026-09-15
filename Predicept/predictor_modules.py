import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    
class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output
    
class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        output = self.position_encode(self.point_net(input))
        encoding, mask = self.segment_map(input, output)

        return encoding, mask
    
class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 256))

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-6)).unsqueeze(-1)
        trajs = torch.cat([trajs, theta, v], dim=-1) # (x, y, heading, vx, vy)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        output = torch.max(trajs, dim=-2).values

        return output

class GMMPredictor(nn.Module):
    def __init__(self, modalities=6):
        super(GMMPredictor, self).__init__()
        self.modalities = modalities
        self._future_len = 80
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 1))
    
    def forward(self, input):
        B, N, M, _ = input.shape
        traj = self.gaussian(input).view(B, N, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return traj, score

class SelfTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

class CrossTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None, attn_bias=None):
        attention_output, _ = self.cross_attention(query, key, value,
                                                   key_padding_mask=mask, attn_mask=attn_bias)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

class InitialPredictionDecoder(nn.Module):
    def __init__(self, modalities, neighbors, dim=256):
        super(InitialPredictionDecoder, self).__init__()
        self._modalities = modalities
        self._agents = neighbors + 1
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(self._agents, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor()
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(self._agents).long())

    def forward(self, current_states, encoding, mask):
        B, N, _ = current_states.shape
        N = self._agents
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent)
        query = encoding[:, :N, None] + multi_modal_query[None, :, :] + agent_query[:, None, :]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask) for i in range(N)], dim=1)
        predictions, scores = self.predictor(query_content)
        predictions[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, predictions, scores

class InteractionDecoder(nn.Module):
    def __init__(self, modalities, future_encoder):
        super(InteractionDecoder, self).__init__()
        self.modalities = modalities
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor()

    def forward(self, current_states, actors, scores, last_content, encoding, mask):
        N = actors.shape[1]
        
        # using future encoder to encode the future trajectories
        multi_futures = self.future_encoder(actors[..., :2], current_states[:, :N])
        
        # using scores to weight the encoded futures
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)    
        
        # using self-attention to encode the interaction
        interaction = self.interaction_encoder(futures, mask[:, :N])
        
        # append the interaction encoding to the common content
        encoding = torch.cat([interaction, encoding], dim=1)

        # mask out the corresponding agents
        mask = torch.cat([mask[:, :N], mask], dim=1)
        mask = mask.unsqueeze(1).expand(-1, N, -1).clone()
        for i in range(N):
            mask[:, i, i] = 1

        # using cross-attention to decode the future trajectories
        query = last_content + multi_futures
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask[:, i]) for i in range(N)], dim=1)
        trajectories, scores = self.decoder(query_content)
        
        # add the current states to the trajectories
        trajectories[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, trajectories, scores  

class PhysicsBasedDecoderv2(nn.Module):
    def __init__(self, modalities=6, future_len=80, dt=0.1):
        super(PhysicsBasedDecoderv2, self).__init__()
        self.modalities = modalities
        self.future_len = future_len
        self.dt = dt
        self.future_encoder = FutureEncoder()

    def get_multimodal_physics_trajectories(self, current_states, map_lanes):
        B, N, D = current_states.shape
        _, L, P, _ = map_lanes.shape
        
        vx = current_states[..., 3]
        vy = current_states[..., 4]
        speed = torch.sqrt(vx**2 + vy**2).unsqueeze(-1) # [B, N, 1]
        
        agent_pos = current_states[..., :2].unsqueeze(2).unsqueeze(2) # [B, N, 1, 1, 2]
        lane_points = map_lanes[..., :2].unsqueeze(1) # [B, 1, L, P, 2]
        dists = torch.norm(agent_pos - lane_points, dim=-1) # [B, N, L, P]
        
        lane_dists, closest_point_idx = torch.min(dists, dim=-1) # [B, N, L]
        
        K = min(3, L)
        topk_dists, topk_lane_idxs = torch.topk(lane_dists, k=K, dim=-1, largest=False) # [B, N, K]
        topk_point_idxs = torch.gather(closest_point_idx, 2, topk_lane_idxs) # [B, N, K]
        
        if K < 3:
            pad_lane = topk_lane_idxs[..., -1:]
            pad_point = topk_point_idxs[..., -1:]
            topk_lane_idxs = torch.cat([topk_lane_idxs, pad_lane.repeat(1, 1, 3-K)], dim=-1)
            topk_point_idxs = torch.cat([topk_point_idxs, pad_point.repeat(1, 1, 3-K)], dim=-1)
            
        flat_batch_idx = torch.arange(B, device=current_states.device).view(B, 1, 1) * L
        flat_lane_idxs = (flat_batch_idx + topk_lane_idxs).view(B * N * 3)
        flat_map_lanes = map_lanes[..., :2].view(B * L, P, 2)
        selected_lanes = flat_map_lanes[flat_lane_idxs].view(B, N, 3, P, 2) # [B, N, 3, P, 2]
        
        diffs = torch.norm(torch.diff(selected_lanes, dim=3, prepend=selected_lanes[:, :, :, :1, :]), dim=-1)
        lane_arc_lengths = torch.cumsum(diffs, dim=-1) # [B, N, 3, P]
        curr_arc_lens = torch.gather(lane_arc_lengths, 3, topk_point_idxs.unsqueeze(-1)).squeeze(-1) # [B, N, 3]
        
        time_steps = torch.arange(1, self.future_len + 1, device=current_states.device).float() * self.dt
        t = time_steps.view(1, 1, -1)
        
        target_arc_lens = torch.zeros((B, N, 5, self.future_len), device=current_states.device)
        target_arc_lens[:, :, 0, :] = curr_arc_lens[:, :, 0:1] + speed * t
        target_arc_lens[:, :, 1, :] = curr_arc_lens[:, :, 0:1] + (speed * 1.5) * t
        target_arc_lens[:, :, 2, :] = curr_arc_lens[:, :, 0:1] + (speed * 0.5) * t
        target_arc_lens[:, :, 3, :] = curr_arc_lens[:, :, 1:2] + speed * t
        target_arc_lens[:, :, 4, :] = curr_arc_lens[:, :, 2:3] + speed * t
        
        lane_mapping = [0, 0, 0, 1, 2]
        trajectories = []
        B_N = B * N
        batch_indices = torch.arange(B_N, device=current_states.device).unsqueeze(1)
        
        for mode in range(5):
            lane_idx = lane_mapping[mode]
            arc_lens = lane_arc_lengths[:, :, lane_idx, :] # [B, N, P]
            targets = target_arc_lens[:, :, mode, :] # [B, N, T]
            
            arc_lens_fixed = arc_lens.clone()
            idx = torch.searchsorted(arc_lens_fixed, targets.clamp(max=arc_lens_fixed[..., -1].unsqueeze(-1)))
            idx = idx.clamp(0, P - 1)
            
            lanes_flat_2d = selected_lanes[:, :, lane_idx, :, :].reshape(B_N, P, 2)
            idx_flat = idx.reshape(B_N, self.future_len)
            
            mode_traj = lanes_flat_2d[batch_indices, idx_flat].reshape(B, N, self.future_len, 2)
            trajectories.append(mode_traj)
            
        delta_x = current_states[..., 3:4] * t
        delta_y = current_states[..., 4:5] * t
        cv_traj = torch.stack([delta_x, delta_y], dim=-1) + current_states[..., :2].unsqueeze(2)
        trajectories.append(cv_traj)
        
        final_trajectories = torch.stack(trajectories, dim=2) # [B, N, 6, T, 2]
        
        return final_trajectories

    def forward(self, current_states, encoding, mask, map_lanes):
        B, N, _ = current_states.shape
        
        predictions_traj_xy = self.get_multimodal_physics_trajectories(current_states, map_lanes)
        
        dummy_log_std = torch.zeros_like(predictions_traj_xy) 
        predictions = torch.cat([predictions_traj_xy, dummy_log_std], dim=-1) 
        
        encoded_futures = self.future_encoder(predictions[..., :2], current_states)
        
        query_content = encoded_futures # [B, N, M, 256]
        
        scores = torch.zeros((B, N, self.modalities), device=current_states.device)
        
        return query_content, predictions, scores
    

    def __init__(self, d_model=256, num_lat=5, num_lon=12, max_speed=15.0):
        super(PhysicalModeEmbedding, self).__init__()
        self.d_model = d_model
        self.num_lat = num_lat
        self.num_lon = num_lon
        self.max_speed = max_speed
        
        self.pointnet = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        
        self.mode_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

    def forward(self, c_lat_candidates):
        B, N_lat, N_pts, D_in = c_lat_candidates.shape
        device = c_lat_candidates.device
        
        c_lat_flat = c_lat_candidates.view(B * N_lat, N_pts, D_in).permute(0, 2, 1).float()
        lat_features = self.pointnet(c_lat_flat)
        lat_features = torch.max(lat_features, dim=2)[0]
        lat_features = lat_features.view(B, N_lat, self.d_model)
        
        speeds = torch.linspace(0, self.max_speed, self.num_lon, device=device).unsqueeze(1)
        lon_features = self.speed_mlp(speeds)
        
        lat_expanded = lat_features.unsqueeze(2).expand(B, self.num_lat, self.num_lon, self.d_model)
        lon_expanded = lon_features.unsqueeze(0).unsqueeze(0).expand(B, self.num_lat, self.num_lon, self.d_model)
        
        combined_features = torch.cat([lat_expanded, lon_expanded], dim=-1)
        fused_modes = self.mode_fusion(combined_features)
        
        return fused_modes.view(B, self.num_lat * self.num_lon, self.d_model)
    

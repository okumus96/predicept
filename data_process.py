import os
import yaml
import argparse
from tqdm import tqdm
from common_utils import *
from Predicept.data_utils import *
import matplotlib.pyplot as plt
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.common.actor_state.state_representation import Point2D
import math
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from shapely import Point
from Planner.state_lattice_path_planner import LatticePlanner
import numpy as np
from Planner.planner_utils import annotate_occupancy


# define data processor
class DataProcessor(object):
    def __init__(self, scenarios):
        self._scenarios = scenarios

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20

        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 60 # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

    def get_multimodal_reference_paths(self, ego_state, traffic_light_data, max_routes=5, points_per_route=1500, search_distance=150.0):
        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_heading = ego_state.rear_axle.heading

        c_lat_candidates_ego = np.zeros((max_routes, points_per_route, 6), dtype=np.float32)
        c_lat_candidates_global = np.zeros((max_routes, points_per_route, 6), dtype=np.float32)

        route_blocks = []
        route_edge_ids = []
        route_roadblock_ids = self.scenario.get_route_roadblock_ids() if hasattr(self, 'scenario') else []
        for id_ in route_roadblock_ids:
            block = self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if not block:
                continue
            route_blocks.append(block)
            for edge in block.interior_edges:
                route_edge_ids.append(edge.id)

        if not route_blocks or not route_edge_ids:
            return c_lat_candidates_ego, c_lat_candidates_global

        cur_point = (ego_x, ego_y)
        closest_distance = np.inf
        starting_block = None
        for block in route_blocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    closest_distance = distance
                    starting_block = block

            if np.isclose(closest_distance, 0):
                break

        if starting_block is None:
            return c_lat_candidates_ego, c_lat_candidates_global

        lattice_planner = LatticePlanner(route_edge_ids, max_len=search_distance)
        edges = lattice_planner.get_candidate_edges(starting_block, ego_state)
        candidate_paths = lattice_planner.get_candidate_paths(edges)
        if candidate_paths is None:
            return c_lat_candidates_ego, c_lat_candidates_global

        target_speed = starting_block.interior_edges[0].speed_limit_mps
        if target_speed is None:
            target_speed = 15.0

        c_rot, s_rot = np.cos(-ego_heading), np.sin(-ego_heading)

        for i, (_, (_, _, lane_path, path_polyline)) in enumerate(candidate_paths):
            if i >= max_routes:
                break

            lane_ids = set([str(l.id) for l in lane_path])

            full_centerline_ego = []
            full_centerline_global = []
            for p in path_polyline:
                dx = p[0] - ego_x
                dy = p[1] - ego_y
                rel_x = dx * c_rot - dy * s_rot
                rel_y = dx * s_rot + dy * c_rot
                rel_yaw = p[2] - ego_heading

                if rel_x > -2.0:
                    full_centerline_ego.append([rel_x, rel_y, rel_yaw])
                    full_centerline_global.append([p[0], p[1], p[2]])

            full_centerline_ego = np.array(full_centerline_ego)
            full_centerline_global = np.array(full_centerline_global)
            if len(full_centerline_ego) <= 1:
                continue

            orig_indices = np.linspace(0, 1, len(full_centerline_ego))
            target_indices = np.linspace(0, 1, points_per_route)
            ref_x_ego = np.interp(target_indices, orig_indices, full_centerline_ego[:, 0])
            ref_y_ego = np.interp(target_indices, orig_indices, full_centerline_ego[:, 1])
            ref_yaw_ego = np.interp(target_indices, orig_indices, full_centerline_ego[:, 2])

            ref_x_global = np.interp(target_indices, orig_indices, full_centerline_global[:, 0])
            ref_y_global = np.interp(target_indices, orig_indices, full_centerline_global[:, 1])
            ref_yaw_global = np.interp(target_indices, orig_indices, full_centerline_global[:, 2])

            dx = np.gradient(ref_x_global)
            dy = np.gradient(ref_y_global)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

            max_speed = np.full(points_per_route, target_speed, dtype=np.float32)

            occupancy = np.zeros(points_per_route)
            for data in traffic_light_data:
                if data.status == TrafficLightStatusType.RED and str(data.lane_connector_id) in lane_ids:
                    lane_conn = self.map_api.get_map_object(str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
                    if not lane_conn:
                        continue

                    conn_path = lane_conn.baseline_path.discrete_path
                    if len(conn_path) == 0:
                        continue

                    cx = conn_path[0].x - ego_x
                    cy = conn_path[0].y - ego_y
                    rel_cx = cx * c_rot - cy * s_rot
                    occupancy[ref_x_ego > rel_cx] = 1.0

            c_lat_candidates_ego[i] = np.stack([ref_x_ego, ref_y_ego, ref_yaw_ego, curvature, max_speed, occupancy], axis=-1)
            c_lat_candidates_global[i] = np.stack([ref_x_global, ref_y_global, ref_yaw_global, curvature, max_speed, occupancy], axis=-1)


        return c_lat_candidates_ego, c_lat_candidates_global

    def get_multimodal_reference_paths2(self, ego_state, traffic_light_data, max_routes=5, points_per_route=1500, search_distance=150.0):
        import numpy as np
        import scipy.spatial
        from shapely import Point
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType

        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_heading = ego_state.rear_axle.heading

        c_lat_candidates_ego = np.zeros((max_routes, points_per_route, 6), dtype=np.float32)
        c_lat_candidates_global = np.zeros((max_routes, points_per_route, 6), dtype=np.float32)

        route_blocks = []
        route_edge_ids = []
        route_roadblock_ids = self.scenario.get_route_roadblock_ids() if hasattr(self, 'scenario') else []
        for id_ in route_roadblock_ids:
            block = self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if not block:
                continue
            route_blocks.append(block)
            for edge in block.interior_edges:
                route_edge_ids.append(edge.id)

        if not route_blocks or not route_edge_ids:
            return c_lat_candidates_ego, c_lat_candidates_global

        cur_point = (ego_x, ego_y)
        closest_distance = np.inf
        starting_block = None
        for block in route_blocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    closest_distance = distance
                    starting_block = block

            if np.isclose(closest_distance, 0):
                break

        if starting_block is None:
            return c_lat_candidates_ego, c_lat_candidates_global

        lattice_planner = LatticePlanner(route_edge_ids, max_len=search_distance)
        edges = lattice_planner.get_candidate_edges(starting_block, ego_state)
        candidate_paths = lattice_planner.get_candidate_paths(edges)
        if candidate_paths is None:
            return c_lat_candidates_ego, c_lat_candidates_global

        target_speed = starting_block.interior_edges[0].speed_limit_mps
        if target_speed is None:
            target_speed = 15.0

        c_rot, s_rot = np.cos(-ego_heading), np.sin(-ego_heading)
        
        valid_count = 0

        for _, (_, _, lane_path, path_polyline) in candidate_paths:
            if valid_count >= max_routes:
                break

            lane_ids = set([str(l.id) for l in lane_path])

            full_centerline_ego = []
            full_centerline_global = []
            for p in path_polyline:
                dx = p[0] - ego_x
                dy = p[1] - ego_y
                rel_x = dx * c_rot - dy * s_rot
                rel_y = dx * s_rot + dy * c_rot
                rel_yaw = p[2] - ego_heading

                if rel_x > -2.0:
                    full_centerline_ego.append([rel_x, rel_y, rel_yaw])
                    full_centerline_global.append([p[0], p[1], p[2]])

            full_centerline_ego = np.array(full_centerline_ego)
            full_centerline_global = np.array(full_centerline_global)
            if len(full_centerline_ego) <= 1:
                continue

            dp = np.diff(full_centerline_ego, axis=0)
            segment_dists = np.hypot(dp[:, 0], dp[:, 1])
            cum_dists = np.insert(np.cumsum(segment_dists), 0, 0.0)
            
            target_dists = np.linspace(0, cum_dists[-1], points_per_route)
            
            ref_x_ego = np.interp(target_dists, cum_dists, full_centerline_ego[:, 0])
            ref_y_ego = np.interp(target_dists, cum_dists, full_centerline_ego[:, 1])
            ref_yaw_ego = np.interp(target_dists, cum_dists, full_centerline_ego[:, 2])

            ref_x_global = np.interp(target_dists, cum_dists, full_centerline_global[:, 0])
            ref_y_global = np.interp(target_dists, cum_dists, full_centerline_global[:, 1])
            ref_yaw_global = np.interp(target_dists, cum_dists, full_centerline_global[:, 2])

            dx = np.gradient(ref_x_global)
            dy = np.gradient(ref_y_global)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

            max_speed = np.full(points_per_route, target_speed, dtype=np.float32)

            occupancy = np.zeros((points_per_route, 1))
            ego_path_2d = np.column_stack([ref_x_ego, ref_y_ego])
            
            for data in traffic_light_data:
                if data.status == TrafficLightStatusType.RED and str(data.lane_connector_id) in lane_ids:
                    lane_conn = self.map_api.get_map_object(str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
                    if not lane_conn:
                        continue

                    conn_path = lane_conn.baseline_path.discrete_path
                    if len(conn_path) == 0:
                        continue

                    conn_pts = np.array([[p.x, p.y] for p in conn_path])
                    dx_l = conn_pts[:, 0] - ego_x
                    dy_l = conn_pts[:, 1] - ego_y
                    rel_cx = dx_l * c_rot - dy_l * s_rot
                    rel_cy = dx_l * s_rot + dy_l * c_rot
                    red_light_lane_ego = np.column_stack([rel_cx, rel_cy])
                    
                    occupancy = annotate_occupancy(occupancy, ego_path_2d, red_light_lane_ego)

            occupancy_flat = occupancy.squeeze()

            c_lat_candidates_ego[valid_count] = np.stack([ref_x_ego, ref_y_ego, ref_yaw_ego, curvature, max_speed, occupancy_flat], axis=-1)
            c_lat_candidates_global[valid_count] = np.stack([ref_x_global, ref_y_global, ref_yaw_global, curvature, max_speed, occupancy_flat], axis=-1)
            
            valid_count += 1

        
        return c_lat_candidates_ego, c_lat_candidates_global
    

    def visualize_bfs_candidates(self, ego_past, ego_future, c_lat_candidates, map_name, token):


        plt.figure(figsize=(12, 12))
        
        plt.plot(ego_past[:, 0], ego_past[:, 1], color='black', linewidth=2, label='Ego Past')
        
        current_x, current_y = ego_past[-1, 0], ego_past[-1, 1]
        plt.scatter(current_x, current_y, c='red', s=100, marker='s', zorder=5, label='Ego Current State')

        colors = ['blue', 'orange', 'purple', 'cyan', 'magenta']
        for i in range(c_lat_candidates.shape[0]):
            route = c_lat_candidates[i]
            if np.any(route): 
                plt.plot(route[:, 0], route[:, 1], color=colors[i], linestyle='--', 
                         linewidth=2.5, alpha=0.8, label=f'C_lat Candidate {i}')
                plt.scatter(route[-1, 0], route[-1, 1], color=colors[i], s=50, marker='x', zorder=4)

        plt.plot(ego_future[:, 0], ego_future[:, 1], color='limegreen', linewidth=4, 
                 zorder=3, label='Expert Future (Ground Truth)')

        plt.title(f"BFS Mode Candidates\nMap: {map_name} | Token: {token}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        plt.show()

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = list(self.scenario.get_traffic_light_status_at_iteration(0))

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        lanes = vector_map['lanes']
        vector_map['lane_tl'] = lanes[..., 3:7].copy()
        red = lanes[..., 5] > 0.5
        lanes[..., 5][red] = 0.0
        lanes[..., 6][red] = 1.0
        vector_map['lanes'] = lanes

        polys = self._query_polygon_layers(
            ego_coords, [SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE],
            ego_state.rear_axle)
        vector_map['intersections'] = polys[SemanticMapLayer.INTERSECTION]
        vector_map['stop_polygons'] = polys[SemanticMapLayer.STOP_LINE]
        return vector_map

    def _query_polygon_layers(self, point, layers, anchor, max_elems=20, max_pts=20):
        outs = {layer: np.zeros((max_elems, max_pts, 3), dtype=np.float32) for layer in layers}
        try:
            got = self.map_api.get_proximal_map_objects(point, self._radius, layers)
        except AssertionError as e:
            print(f"[warn] unsupported map layer: {[l.name for l in layers]} -- {e}")
            return outs
        except Exception:
            return outs
        ca, sa = math.cos(anchor.heading), math.sin(anchor.heading)
        for layer in layers:
            out = outs[layer]
            k = 0
            for obj in got.get(layer, []):
                if k >= max_elems:
                    break
                try:
                    xy = np.array(obj.polygon.exterior.coords, dtype=np.float32)
                except Exception:
                    continue
                if len(xy) < 2:
                    continue
                if len(xy) > max_pts:
                    xy = xy[np.linspace(0, len(xy) - 1, max_pts).astype(int)]
                dx, dy = xy[:, 0] - anchor.x, xy[:, 1] - anchor.y
                out[k, :len(xy), 0] = dx * ca + dy * sa
                out[k, :len(xy), 1] = -dx * sa + dy * ca
                out[k, :len(xy), 2] = 1.0
                k += 1
        return outs

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    
    def plot_scenario(self, data):
        # Create map layers
        create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'])

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        create_agents_raster(data['neighbor_agents_past'][:, -1])

        # Draw past and future trajectories
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'])

        if 'c_lat_candidates' in data:
            colors = ['blue', 'orange', 'purple', 'cyan', 'magenta']
            for i, route in enumerate(data['c_lat_candidates']):
                if np.any(route):
                    plt.plot(route[:, 0], route[:, 1], color=colors[i], linestyle='--', linewidth=3, zorder=5)

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def work(self, save_dir, debug=False):
        todo = [sc for sc in self._scenarios
                if not os.path.exists(f"{save_dir}/{sc._map_name}_{sc.token}.npz")]
        print(f"[resume] skipped {len(self._scenarios) - len(todo)} already processed, "
              f"{len(todo)} scenarios left to process")
        for scenario in tqdm(todo):
            map_name = scenario._map_name
            token = scenario.token
            self.scenario = scenario
            self.map_api = scenario.map_api        

            # get agent past tracks
            ego_agent_past, time_stamps_past = self.get_ego_agent()
            neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
            ego_agent_past, neighbor_agents_past, neighbor_indices = \
                agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents)

            # get vector set map
            vector_map = self.get_map()

            # get agent future tracks
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            traffic_light_data = list(self.scenario.get_traffic_light_status_at_iteration(0))
            c_lat_candidates, c_lat_candidates_global = self.get_multimodal_reference_paths2(
                self.scenario.initial_ego_state,
                traffic_light_data,
                max_routes=5,
                points_per_route=1200,
                search_distance=120.0,
            )

            # gather data
            data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future,
                    "c_lat_candidates": c_lat_candidates, "c_lat_candidates_global": c_lat_candidates_global}
            data.update(vector_map)

            # visualization
            if debug:
                self.visualize_bfs_candidates(ego_agent_past, ego_agent_future, c_lat_candidates, map_name, token)
                self.plot_scenario(data)

            # save to disk
            self.save_to_disk(save_dir, data)

def scenario_filter_helper(config_path):
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        scenarios_per_type = config.get('num_scenarios_per_type')
        total_scenarios = config.get('limit_total_scenarios')
        shuffle_scenarios = config.get('shuffle', False)
        scenario_types = config.get('scenario_types')
        scenario_tokens = config.get('scenario_tokens')
        log_names = config.get('log_names')
        map_names = config.get('map_names')
        timestamp_threshold_s = config.get('timestamp_threshold_s')
        ego_displacement_minimum_m = config.get('ego_displacement_minimum_m')
        expand_scenarios = config.get('expand_scenarios')
        remove_invalid_goals = config.get('remove_invalid_goals')
        ego_start_speed_threshold = config.get('ego_start_speed_threshold')
        ego_stop_speed_threshold = config.get('ego_stop_speed_threshold')
        speed_noise_tolerance = config.get('speed_noise_tolerance')
        print(f"Loaded config from: {config_path}")
    else:
        scenarios_per_type = args.scenarios_per_type
        total_scenarios = args.total_scenarios
        shuffle_scenarios = args.shuffle_scenarios
        scenario_types = None
        scenario_tokens = None
        log_names = None
        map_names = None
        timestamp_threshold_s = None
        ego_displacement_minimum_m = None
        expand_scenarios = None
        remove_invalid_goals = None
        ego_start_speed_threshold = None
        ego_stop_speed_threshold = None
        speed_noise_tolerance = None
    
    
    filter_params_default = list(get_filter_parameters(scenarios_per_type, total_scenarios, shuffle_scenarios))
    
    if scenario_types is not None:
        filter_params_default[0] = scenario_types
    else: 
        filter_params_default[0] = None
    if scenario_tokens is not None:
        filter_params_default[1] = scenario_tokens
    if log_names is not None:
        filter_params_default[2] = log_names
    if map_names is not None:
        filter_params_default[3] = map_names
    if timestamp_threshold_s is not None:
        filter_params_default[6] = timestamp_threshold_s
    if ego_displacement_minimum_m is not None:
        filter_params_default[7] = ego_displacement_minimum_m
    if expand_scenarios is not None:
        filter_params_default[8] = expand_scenarios
    if remove_invalid_goals is not None:
        filter_params_default[9] = remove_invalid_goals
    if ego_start_speed_threshold is not None:
        filter_params_default[11] = ego_start_speed_threshold
    if ego_stop_speed_threshold is not None:
        filter_params_default[12] = ego_stop_speed_threshold
    if speed_noise_tolerance is not None:
        filter_params_default[13] = speed_noise_tolerance

    return filter_params_default
    
def main(args):
    # get scenarios
    map_version = "nuplan-maps-v1.0"    
    sensor_root = None
    db_files = None
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    filter_params_default_train = scenario_filter_helper(args.train_config)
    filter_params_default_val = scenario_filter_helper(args.val_config)

    # create save folder
    os.makedirs(args.train_save_path, exist_ok=True)
    os.makedirs(args.val_save_path, exist_ok=True)

    do_val = args.only in ('val', 'both')
    do_train = args.only in ('train', 'both')

    scenarios_val = scenarios_train = None
    if do_val:
        scenarios_val = builder.get_scenarios(ScenarioFilter(*filter_params_default_val), worker)
    if do_train:
        scenarios_train = builder.get_scenarios(ScenarioFilter(*filter_params_default_train), worker)

    del worker, builder, scenario_mapping
    if do_val:
        print(f"[val] Total number of scenarios: {len(scenarios_val)}")
        DataProcessor(scenarios_val).work(args.val_save_path, debug=args.debug)
    if do_train:
        print(f"[train] Total number of scenarios: {len(scenarios_train)}")
        DataProcessor(scenarios_train).work(args.train_save_path, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--val_config', type=str, help='path to scenario filter config YAML for validation dataset')
    parser.add_argument('--train_config', type=str, help='path to scenario filter config YAML for training dataset')
    parser.add_argument('--data_path', type=str, help='path to raw data')
    parser.add_argument('--map_path', type=str, help='path to map data')
    parser.add_argument('--train_save_path', type=str, help='path to save processed data')
    parser.add_argument('--val_save_path', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=4000, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', default=None, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=False, help='shuffle scenarios')
    parser.add_argument('--only', type=str, default='train', choices=['train','val','both'],
                        help='which split to process. val = validation only (quick test)')
    parser.add_argument('--debug', action="store_true", help='if visualize the data output', default=False)
    args = parser.parse_args()

    main(args)


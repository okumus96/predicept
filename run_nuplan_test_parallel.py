import os
import time
import yaml
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from common_utils import *
from Planner.causal_refiner_planner import CausalRefinerPlanner

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile


def build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir):
    """
    Builds the main experiment folder for simulation.
    :return: The main experiment folder path.
    """
    print('Building experiment folders...')

    exp_folder = pathlib.Path(output_dir)
    print(f'\nFolder where all results are stored: {exp_folder}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Build nuboard event file.
    nuboard_filename = exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
    nuboard_file = NuBoardFile(
        simulation_main_path=str(exp_folder),
        simulation_folder=simulation_dir,
        metric_main_path=str(exp_folder),
        metric_folder=metric_dir,
        aggregator_metric_folder=aggregator_metric_dir,
    )

    metric_main_path = exp_folder / metric_dir
    metric_main_path.mkdir(parents=True, exist_ok=True)

    nuboard_file.save_nuboard_file(nuboard_filename)
    print('Building experiment folders...DONE!')

    return exp_folder.name


def build_simulation(experiment, planner, scenarios, output_dir, simulation_dir, metric_dir):
    runner_reports = []
    print(f'Building simulations from {len(scenarios)} scenarios...')

    metric_engine = build_metrics_engine(experiment, output_dir, metric_dir)
    print('Building metric engines...DONE\n')

    # Iterate through scenarios
    for scenario in tqdm(scenarios, desc='Running simulation'):
        tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0],
                             r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10,
                             jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                             stopping_proportional_gain=0.5, stopping_velocity=0.2)
        motion_model = KinematicBicycleModel(get_pacifica_parameters())

        # Ego Controller and Perception
        if experiment == 'open_loop_boxes':
            ego_controller = LogPlaybackController(scenario)
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_nonreactive_agents':
            ego_controller = TwoStageController(scenario, tracker, motion_model)
            observations = TracksObservation(scenario)
        else:
            ego_controller = TwoStageController(scenario, tracker, motion_model)
            observations = IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                     accel_max=1.0, decel_max=2.0, scenario=scenario,
                                     open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN",
                                                                 "TRAFFIC_CONE", "GENERIC_OBJECT"])

        # Simulation Manager
        simulation_time_controller = StepSimulationTimeController(scenario)

        # Stateful callbacks
        metric_callback = MetricCallback(metric_engine=metric_engine)
        sim_log_callback = SimulationLogCallback(output_dir, simulation_dir, "msgpack")

        # Construct simulation and manager
        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )

        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([metric_callback, sim_log_callback])
        )

        # Begin simulation
        simulation_runner = SimulationRunner(simulation, planner)
        report = simulation_runner.run()
        runner_reports.append(report)

    # save reports
    save_runner_reports(runner_reports, output_dir, 'runner_reports')

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0

    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            print("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = len(scenarios) - number_of_successful
    print(f"Number of successful simulations: {number_of_successful}")
    print(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        print(f"Failed simulations [log, token]:\n{failed_simulations}")

    print('Finished running simulations!')

    return runner_reports


def build_nuboard(scenario_builder, simulation_path):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
    )

    nuboard.run()


def main(args):
    # parameters
    experiment_name = args.experiment_name
    job_name = 'predicept'
    experiment_time = datetime.datetime.now()
    shard_tag = ("_s" + args.shard.replace("/", "of")) if args.shard else ""
    tag = f"{shard_tag}_{args.out_tag}" if args.out_tag else shard_tag
    experiment = f"{experiment_name}/{job_name}/{experiment_time}{tag}"
    output_dir = f"testing_log/{experiment}"
    simulation_dir = "simulation"
    metric_dir = "metrics"
    aggregator_metric_dir = "aggregator_metric"

    # initialize planner
    planner = CausalRefinerPlanner(backbone_path=args.model_path, causal_path=args.causal_path,
                                   psi_prior_alpha=args.psi_prior_alpha, device=args.device)

    # initialize main aggregator
    metric_aggregators = build_metrics_aggregators(experiment_name, output_dir, aggregator_metric_dir)
    metric_save_path = f"{output_dir}/{metric_dir}"
    metric_aggregator_callback = MetricAggregatorCallback(metric_save_path, metric_aggregators)
    metric_file_callback = MetricFileCallback(metric_file_output_path=f"{output_dir}/{metric_dir}",
                                              scenario_metric_paths=[f"{output_dir}/{metric_dir}"],
                                              delete_scenario_metric_files=True)
    metric_summary_callback = MetricSummaryCallback(metric_save_path=f"{output_dir}/{metric_dir}",
                                                    metric_aggregator_save_path=f"{output_dir}/{aggregator_metric_dir}",
                                                    summary_output_path=f"{output_dir}/summary",
                                                    num_bins=20, pdf_file_name='summary.pdf')
    main_callbacks = MultiMainCallback([metric_file_callback, metric_aggregator_callback, metric_summary_callback])
    main_callbacks.on_run_simulation_start()

    # build simulation folder
    build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir)

    # build scenarios
    print('Extracting scenarios...')
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version,
                                    scenario_mapping=scenario_mapping)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    filter_params = list(get_filter_parameters(config.get('num_scenarios_per_type'),
                                               config.get('limit_total_scenarios'),
                                               config.get('shuffle', False)))
    overrides = {0: 'scenario_types', 1: 'scenario_tokens', 2: 'log_names', 3: 'map_names',
                 6: 'timestamp_threshold_s', 7: 'ego_displacement_minimum_m', 8: 'expand_scenarios',
                 9: 'remove_invalid_goals', 11: 'ego_start_speed_threshold', 12: 'ego_stop_speed_threshold',
                 13: 'speed_noise_tolerance'}
    for index, key in overrides.items():
        if config.get(key) is not None:
            filter_params[index] = config.get(key)
    print(f"Loaded config from: {args.config}")

    scenario_filter = ScenarioFilter(*filter_params)
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    del worker, scenario_filter, scenario_mapping

    # optional sharding: run only the k-th of n token-sorted slices
    if args.shard:
        k, n = map(int, args.shard.split('/'))
        assert 0 <= k < n, f'invalid shard {args.shard}'
        scenarios = sorted(scenarios, key=lambda sc: sc.token)[k::n]
        print(f'[shard] {k}/{n}: {len(scenarios)} scenarios', flush=True)

    if len(scenarios) == 0:
        print(f"No scenarios found for config {args.config}")
        return

    # begin testing
    build_simulation(experiment_name, planner, scenarios, output_dir, simulation_dir, metric_dir)
    main_callbacks.on_run_simulation_end()
    print(f"Results: {output_dir}")

    # show metrics and scenarios
    if args.nuboard:
        simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir()
                           if file.is_file() and file.suffix == '.nuboard']
        build_nuboard(builder, simulation_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Predicept on nuPlan')
    parser.add_argument('--experiment_name', required=True,
                        choices=['open_loop_boxes', 'closed_loop_nonreactive_agents', 'closed_loop_reactive_agents'])
    parser.add_argument('--config', required=True, help='scenario filter YAML, e.g. config/test14-hard.yaml')
    parser.add_argument('--model_path', required=True, help='frozen GameFormer backbone checkpoint')
    parser.add_argument('--causal_path', required=True, help='Predicept checkpoint')
    parser.add_argument('--data_path', default=os.path.join(os.environ.get('NUPLAN_DATA_ROOT', ''), 'nuplan-v1.1/splits/test'),
                        help='nuPlan data split (default: $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test)')
    parser.add_argument('--map_path', default=os.environ.get('NUPLAN_MAPS_ROOT', ''),
                        help='nuPlan maps (default: $NUPLAN_MAPS_ROOT)')
    parser.add_argument('--psi_prior_alpha', type=float, default=0.0,
                        help='decision prior correction: 0 for closed loop, 0.75 for open loop')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--shard', type=str, default='', help='k/n: run only the k-th of n scenario shards')
    parser.add_argument('--out_tag', type=str, default='', help='tag appended to the output folder name')
    parser.add_argument('--nuboard', action='store_true', help='launch nuBoard after the simulation')
    args = parser.parse_args()

    main(args)

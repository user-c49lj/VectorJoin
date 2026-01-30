#!/usr/bin/env python3
"""
Experiment runner for simjoin_parallel_profiled.cpp with concurrent task execution.

This script manages a queue of parameter combinations and executes them with controlled parallelism.
"""

import subprocess
import threading
import queue
import time
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
import signal
import logging
import struct

use_reverse_params = False
use_pruning_params = False

def read_fvecs_metadata(filename: str) -> Tuple[int, int]:
    """
    Read metadata from an fvecs file (number of vectors and dimension).

    Args:
        filename (str): Path to the fvecs file

    Returns:
        tuple: (num_vectors, dimension)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Fvecs file not found: {filename}")

    num_vectors = 0
    dimension = None

    with open(filename, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break

            current_dim = struct.unpack('i', dim_bytes)[0]

            # Set dimension on first vector
            if dimension is None:
                dimension = current_dim
            elif current_dim != dimension:
                raise ValueError(f"Inconsistent dimension: expected {dimension}, got {current_dim} at vector {num_vectors}")

            # Read vector data
            vec_bytes = f.read(dimension * 4)
            if len(vec_bytes) < dimension * 4:
                break

            num_vectors += 1

            # Progress indicator for large files
            if num_vectors % 10000 == 0:
                print(f"  Read {num_vectors} vectors from {filename}...")

    print(f"Detected {num_vectors} vectors with dimension {dimension} in {filename}")
    return num_vectors, dimension

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser('~'), 'experiment_runner.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentTask:
    """Represents a single experiment task with parameters and metadata."""

    def __init__(self, task_id: int, params: Dict[str, Any], output_dir: str):
        self.task_id = task_id
        self.params = params
        self.output_dir = output_dir
        self.status = "pending"  # pending, running, completed, failed
        self.start_time = None
        self.end_time = None
        self.return_code = None
        self.stdout = ""
        self.stderr = ""

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'params': self.params,
            'output_dir': self.output_dir,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'return_code': self.return_code,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None
        }

class ExperimentRunner:
    """Manages concurrent execution of experiment tasks."""

    def __init__(self,max_concurrent: int = 4, base_output_dir: str = "experiments", experiment: str = "es_mi_adapt"):
        self.binary_path = './build/ws_join' # Default, can get changed to "./build/naive_join"
        self.max_concurrent = max_concurrent
        self.base_output_dir = base_output_dir
        self.experiment = experiment
        self.task_queue = queue.Queue()
        self.running_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.shutdown_requested = False
        self.results_file = os.path.join(base_output_dir, "experiment_results.json")

        # Create base output directory
        os.makedirs(base_output_dir, exist_ok=True)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def add_task(self, params: Dict[str, Any], custom_output_dir: Optional[str] = None) -> int:
        """Add a task to the queue."""
        task_id = len(self.completed_tasks) + len(self.failed_tasks) + len(self.running_tasks) + self.task_queue.qsize()

        if custom_output_dir:
            output_dir = custom_output_dir
        else:
            # Generate full result directory name using the same logic as C++ programs
            if self.experiment != 'naive':
                result_dir_name = generate_simjoin_result_dir(**params)
            else:
                result_dir_name = generate_simple_result_dir(**params)
            # Prefix with task ID
            output_dir = os.path.join(self.base_output_dir, f"task_{task_id:03d}_{result_dir_name}")

        task = ExperimentTask(task_id, params, output_dir)
        self.task_queue.put(task)
        logger.info(f"Added task {task_id} to queue: {params}")
        return task_id

    def _run_task(self, task: ExperimentTask):
        """Execute a single task."""
        task.status = "running"
        task.start_time = datetime.now()
        self.running_tasks[task.task_id] = task

        logger.info(f"Starting task {task.task_id}: {task.params}")

        try:
            # Create output directory for logs/stdout only
            os.makedirs(task.output_dir, exist_ok=True)

            # Build command
            if self.experiment != 'naive':
                self.binary_path = './build/ws_join'
            else: 
                self.binary_path = './build/naive_join'

            cmd = [self.binary_path]

            # Use the task output directory as the result directory for C++ program
            cmd.append(task.output_dir)

            # Add parameters in the expected order
            # You'll need to adjust this based on your actual parameter order

            if self.experiment != 'naive':
                param_order = [
                    'epsilon', 'w_queue', 'num_threads',
                    'query_index', 'data_index',
                    'query_vectors', 'data_vectors',
                    'prefix', # deprecated
                    'num_queries',
                    'dimension',
                    'no_work_sharing', 'early_stopping_seed', 'early_stopping_greedy', 'soft_work_sharing',
                    'enable_query_to_data_edges', 'k_top_data_points', 'collect_bfs_data', 'break_before_bfs',
                    'seed_offset', 'sort_jj_by_distance', 'enable_seed_offset_filtering', 'cache_closest_only',
                    'jump_threshold', 'epsilon_adapt_factor', 'one_hop_data_only', 
                    'topK', 'patience', 'ood_file','number_cached',
                    'clusters1_file', 'clusters2_file'
                ]
            else:
                param_order = [
                    'epsilon', 'num_threads',
                    'num_queries',
                    'query_vectors', 'data_vectors',
                    'prefix', # deprecated
                    'dimension',
                    'clusters_dir', 'use_anchors'
                ]

            for param in param_order:
                if param in task.params:
                    cmd.append(str(task.params[param]))

            # Run binary in current directory (binary handles its own output)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()  # Run in current directory, not task.output_dir
            )

            # Capture output
            stdout, stderr = process.communicate()

            task.return_code = process.returncode
            task.stdout = stdout
            task.stderr = stderr
            task.end_time = datetime.now()

            # Save stdout/stderr to task output directory
            with open(os.path.join(task.output_dir, 'stdout.txt'), 'w') as f:
                f.write(stdout)
            with open(os.path.join(task.output_dir, 'stderr.txt'), 'w') as f:
                f.write(stderr)

            if task.return_code == 0:
                task.status = "completed"
                self.completed_tasks.append(task)
                logger.info(f"Task {task.task_id} completed successfully in {(task.end_time - task.start_time).total_seconds():.2f}s")
            else:
                task.status = "failed"
                self.failed_tasks.append(task)
                logger.error(f"Task {task.task_id} failed with return code {task.return_code}")
                logger.error(f"STDERR: {stderr}")

        except Exception as e:
            task.status = "failed"
            task.end_time = datetime.now()
            task.stderr = str(e)
            self.failed_tasks.append(task)
            logger.error(f"Task {task.task_id} failed with exception: {e}")

        finally:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

            # Save results
            self._save_results()

    def _save_results(self):
        """Save current results to JSON file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'completed_tasks': [task.to_dict() for task in self.completed_tasks],
            'failed_tasks': [task.to_dict() for task in self.failed_tasks],
            'running_tasks': [task.to_dict() for task in self.running_tasks.values()],
            'queue_size': self.task_queue.qsize()
        }

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def run(self):
        """Main execution loop."""
        logger.info(f"Starting experiment runner with {self.max_concurrent} max concurrent tasks")
        logger.info(f"Binary path: {self.binary_path}")
        logger.info(f"Base output directory: {self.base_output_dir}")

        threads = []

        try:
            while not self.shutdown_requested:
                # Start new tasks if we have capacity and tasks in queue
                while (len(self.running_tasks) < self.max_concurrent and
                       not self.task_queue.empty() and
                       not self.shutdown_requested):

                    try:
                        task = self.task_queue.get(timeout=1)
                        thread = threading.Thread(target=self._run_task, args=(task,))
                        thread.daemon = True
                        thread.start()
                        threads.append(thread)

                    except queue.Empty:
                        break

                # Check if all tasks are done
                if (self.task_queue.empty() and
                    len(self.running_tasks) == 0 and
                    not self.shutdown_requested):
                    logger.info("All tasks completed!")
                    break

                # Wait a bit before checking again
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.shutdown_requested = True

        # Wait for running tasks to complete
        logger.info("Waiting for running tasks to complete...")
        for thread in threads:
            thread.join(timeout=30)  # Wait up to 30 seconds per thread

        # Final results
        logger.info(f"Experiment completed!")
        logger.info(f"Completed tasks: {len(self.completed_tasks)}")
        logger.info(f"Failed tasks: {len(self.failed_tasks)}")
        logger.info(f"Results saved to: {self.results_file}")



def generate_simjoin_result_dir(**kwargs):
    """Generate result directory name for simjoin_parallel_profiled.cpp based on parameters."""
    prefix = kwargs.get('prefix', 'unknown')
    epsilon = kwargs.get('epsilon', 0)
    w_queue = kwargs.get('w_queue', 128)
    num_threads = kwargs.get('num_threads', 1)
    num_queries = kwargs.get('num_queries', 0)
    
    result_dir = f"{prefix}_epsilon_{epsilon}_w_{w_queue}_threads_{num_threads}_Q{num_queries}"
    
    if kwargs.get('no_work_sharing', '0') == '1':
        result_dir += "_NOWS"
    if kwargs.get('early_stopping_seed', '0') == '1':
        result_dir += "_ESS"
    if kwargs.get('early_stopping_greedy', '0') == '1':
        result_dir += "_ESN"
    if kwargs.get('soft_work_sharing', '0') == '1':
        result_dir += "_SWS"
    if kwargs.get('enable_query_to_data_edges', '0') == '1':
        result_dir += "_QD"
        k_top = kwargs.get('k_top_data_points', '0')
        if k_top != '0':
            result_dir += f"_Top{k_top}"
    if kwargs.get('collect_bfs_data', '0') == '1':
        result_dir += "_BFSC"
    if kwargs.get('break_before_bfs', '0') == '1':
        result_dir += "_BBF"
    seed_offset = kwargs.get('seed_offset', '0')
    if seed_offset != '0':
        result_dir += f"_SO{seed_offset}"
    if kwargs.get('sort_jj_by_distance', '0') == '1':
        result_dir += "_SJ"
    if kwargs.get('enable_seed_offset_filtering', '0') == '1':
        result_dir += "_SOF"
    if kwargs.get('cache_closest_only', '0') == '1':
        result_dir += "_CC"
    jump_threshold = kwargs.get('jump_threshold', '0')
    if jump_threshold != '0':
        result_dir += f"_JUMP{jump_threshold}"
    adapt_factor = kwargs.get('epsilon_adapt_factor', '0')
    if adapt_factor != '0':
        result_dir += f"_ADAPT{adapt_factor}"
    if kwargs.get('one_hop_data_only', '0') == '1':
        result_dir += "_OH"
    if kwargs.get('clusters1_file', ''):
        result_dir += "_C1"
    if kwargs.get('clusters2_file', ''):
        result_dir += "_C2"
    if kwargs.get('topK', ''):
        topk = kwargs.get('topK', '')
        result_dir += f"_TopK{topk}"
    if kwargs.get('patience', ''):
        patience = kwargs.get('patience', '')
        result_dir += f"_PAT{patience}"
    if kwargs.get('number_cached', ''):
        number_cached = kwargs.get('number_cached', '')
        result_dir += f"_NC{number_cached}"

    
    return result_dir

def generate_simple_result_dir(**kwargs):
    """Generate result directory name for simple_simjoin.cpp based on parameters."""
    prefix = kwargs.get('prefix', 'unknown')
    epsilon = kwargs.get('epsilon', 0)
    num_threads = kwargs.get('num_threads', 1)
    num_queries = kwargs.get('num_queries', 0)
    use_anchors = kwargs.get('use_anchors', '0')
    clusters_dir = kwargs.get('clusters_dir', '')
    
    result_dir = f"{prefix}_epsilon_{epsilon}_threads_{num_threads}_Q{num_queries}"
    if clusters_dir:
        result_dir += "_multicluster"
        if use_anchors == '1':
            result_dir += "_anchors"
    return result_dir

def create_parameter_combinations(dataset='sift', prefix=None, experiment='es_mi_adapt'):
    # Note: prefix parameter is deprecated, but we still need it for backward compatibility
    print(f"Creating parameter combinations for dataset: {dataset}...")
    """Create a set of parameter combinations for experiments."""
    data = dataset

    # Define file paths
    query_fvecs = f'{data}/{data}_query.fvecs'
    data_fvecs = f'{data}/{data}_base.fvecs'

    # Auto-detect parameters from fvecs files
    print(f"Auto-detecting parameters from fvecs files...")
    try:
        num_queries, dimension = read_fvecs_metadata(query_fvecs)
        num_data_vectors, data_dimension = read_fvecs_metadata(data_fvecs)

        # Verify dimensions match
        if dimension != data_dimension:
            raise ValueError(f"Dimension mismatch: query vectors have dimension {dimension}, data vectors have dimension {data_dimension}")

        print(f"Auto-detected parameters:")
        print(f"  - Number of queries: {num_queries}")
        print(f"  - Number of data vectors: {num_data_vectors}")
        print(f"  - Dimension: {dimension}")

    except FileNotFoundError as e:
        print(f"Warning: Could not auto-detect parameters: {e}")
        print("Falling back to hardcoded values...")
        assert False
    else:
        # Set epsilon values based on dimension (you can customize this logic)
        if data == 'sift':
            epsilon_values = [50, 100, 150, 200, 250, 300, 350]
        elif data == 'gist':
            epsilon_values = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        elif 'fmnist' in data:
            epsilon_values = [500, 750, 1000, 1250, 1500, 1750, 2000]
        elif data == 'nytimes':
            epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
        elif data == 'glove':
            epsilon_values = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        elif data == 'coco':
            epsilon_values = [1.33, 1.335, 1.34, 1.345, 1.35, 1.355, 1.36]
        elif data == 'imagenet':
            epsilon_values = [1.19, 1.21, 1.23, 1.25, 1.27, 1.29, 1.31]
        elif data == 'laion':
            epsilon_values = [1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24]
        else:
            assert False

    if experiment != 'naive':
        # Base parameters
        base_params = {
            'epsilon': 0,
            'w_queue': 128,
            'num_threads': 1,
            'query_index': f'{data}/{data}_query.nsg',
            'data_index': f'{data}/{data}_base.nsg',
            'query_vectors': f'{data}/{data}_query.fvecs',
            'data_vectors': f'{data}/{data}_base.fvecs',
            'prefix': data + '_sigmod',  # Use dataset-based prefix
            'num_queries': num_queries,
            'dimension': dimension,
            'no_work_sharing': '0',
            'early_stopping_seed': '0',
            'early_stopping_greedy': '0',
            'soft_work_sharing': '0',
            'enable_query_to_data_edges': '0',
            'k_top_data_points': '0',
            'collect_bfs_data': '0',
            'break_before_bfs': '0',
            'seed_offset': '0',
            'sort_jj_by_distance': '0',
            'enable_seed_offset_filtering': '0',
            'cache_closest_only': '0',
            'jump_threshold': '0',
            'epsilon_adapt_factor': '0',
            'one_hop_data_only': '0',
            'topK': '0',
            'patience': '0',
            'ood_file': f'{data}/{data}_ood.txt',
            'number_cached': '0',
            'clusters1_file': '',
            'clusters2_file': ''
        }
        if use_reverse_params:
            # Base parameters
            base_params['query_index'] = f'{data}/{data}_base.nsg'
            base_params['data_index'] = f'{data}/{data}_query.nsg'
            base_params['query_vectors'] = f'{data}/{data}_base.fvecs'
            base_params['data_vectors'] = f'{data}/{data}_query.fvecs'
            base_params['prefix'] = f'{data}_nsg_reverse'
            base_params['num_queries'] = num_data_vectors  # Use auto-detected data vector count
        if use_pruning_params:
            base_params['clusters2_file'] = f'{data}/clusters_1000'
            base_params['prefix'] = f'{data}_pruning_1000'

        combinations = []

        for epsilon in epsilon_values:
            if use_reverse_params:
                index = f'{data}/{data}_all.nsg'


                if experiment == 'index' or experiment == 'all':    
                    # NOWS
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'no_work_sharing': '1',
                    })
                    combinations.append(params)

                if experiment == 'es' or experiment == 'all':
                    # NOWS ES
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'no_work_sharing': '1',
                    })
                    combinations.append(params)

                if experiment == 'es_hws' or experiment == 'all':
                    # HWS
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                    })
                    combinations.append(params)

                if experiment == 'es_sws' or experiment == 'all':
                    # SWS
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'soft_work_sharing': '1',
                        'cache_closest_only': '1',
                    })
                    combinations.append(params)

                if experiment == 'es_mi' or experiment == 'all':
                    # ES + MI
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'no_work_sharing': '1',
                        'seed_offset': num_data_vectors,
                        'data_index': index,
                        'data_vectors': f'{data}/{data}_all.fvecs',
                        'enable_seed_offset_filtering': '1', # added
                        'ood_file': f'{data}/{data}_ood.txt'
                    })
                    combinations.append(params)

                if experiment == 'es_mi_adapt' or experiment == 'all':
                    # MI + Adaptive
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'no_work_sharing': '1',
                        'seed_offset': num_data_vectors,
                        'data_index': index,
                        'data_vectors': f'{data}/{data}_all.fvecs',
                        'enable_seed_offset_filtering': '1', # added
                        'patience': '1',
                        'topK': 256,
                        'ood_file': f'{data}/{data}_ood.txt'
                    })
                    combinations.append(params)
                
            else:
                
                index = f'{data}/{data}_all.nsg'


                if experiment == 'index' or experiment == 'all':    
                    # NOWS
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'no_work_sharing': '1',
                    })
                    combinations.append(params)

                if experiment == 'es' or experiment == 'all':
                    # NOWS ES
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'no_work_sharing': '1',
                    })
                    combinations.append(params)

                if experiment == 'es_hws' or experiment == 'all':
                    # HWS
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                    })
                    combinations.append(params)

                if experiment == 'es_sws' or experiment == 'all':
                    # SWS
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'soft_work_sharing': '1',
                        'cache_closest_only': '1',
                    })
                    combinations.append(params)

                if experiment == 'es_mi' or experiment == 'all':
                    # ES + MI
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'no_work_sharing': '1',
                        'seed_offset': num_data_vectors,
                        'data_index': index,
                        'data_vectors': f'{data}/{data}_all.fvecs',
                        'enable_seed_offset_filtering': '1', # added
                        'ood_file': f'{data}/{data}_ood.txt'
                    })
                    combinations.append(params)

                if experiment == 'es_mi_adapt' or experiment == 'all':
                    # MI + Adaptive
                    params = base_params.copy()
                    params.update({
                        'epsilon': epsilon,
                        'early_stopping_seed': '1',
                        'early_stopping_greedy': '1',
                        'no_work_sharing': '1',
                        'seed_offset': num_data_vectors,
                        'data_index': index,
                        'data_vectors': f'{data}/{data}_all.fvecs',
                        'enable_seed_offset_filtering': '1', # added
                        'patience': '1',
                        'topK': 256,
                        'ood_file': f'{data}/{data}_ood.txt'
                    })
                    combinations.append(params)

    else:
        base_params = {
            'epsilon': 0,
            'num_threads': 1,
            'num_queries': num_queries,
            'query_vectors' : f'{dataset}/{dataset}_query.fvecs',
            'data_vectors' : f'{dataset}/{dataset}_base.fvecs',
            'prefix': dataset + '_test',
            'dimension': dimension,
            'clusters_dir': '',
            'use_anchors': '0',
        }

        combinations = []

        for epsilon in epsilon_values:
            # default mode
            if True:
                params = base_params.copy()
                params.update({
                    'epsilon': epsilon,
                })
                combinations.append(params)

            if False:
                # clustering mode
                params = base_params.copy()
                params.update({
                    'epsilon': epsilon,
                    'clusters_dir': f'{dataset}/clusters_1000',
                    'use_anchors': '0',
                })
                combinations.append(params)

    return combinations

def main():
    parser = argparse.ArgumentParser(description='Run simjoin experiments with concurrent execution')
    parser.add_argument('--max-concurrent', type=int, default=4, help='Maximum number of concurrent tasks')
    parser.add_argument('--output-dir', default='logs', help='Base output directory')
    parser.add_argument('--param-file', help='JSON file with parameter combinations')
    parser.add_argument('--dataset', choices=['sift', 'laion', 'gist', 'fmnist', 'fmnist_con', 'fmnist_con_500', 'fmnist_con_1000', 'fmnist_con_1500', 'fmnist_con_2000', 'fmnist_con_2500', 'nytimes', 'glove','coco','imagenet'], default='sift', help='Dataset to use for experiments')
    parser.add_argument('--use-simjoin-params', action='store_true', default=False, help='Use simjoin parameters (default: True)')
    parser.add_argument('--prefix', help='Prefix for output files (default: dataset-specific)')
    parser.add_argument('--dry-run', action='store_true', help='Print parameter combinations without running')
    parser.add_argument('--experiment', choices=['naive', 'index', 'es', 'es_hws', 'es_sws', 'es_mi', 'es_mi_adapt', 'all'], default='es_mi_adapt', help='The experiment type to run')

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(args.max_concurrent, args.output_dir, args.experiment)

    # Load parameter combinations
    if args.param_file:
        with open(args.param_file, 'r') as f:
            combinations = json.load(f)
    else:
        combinations = create_parameter_combinations(args.dataset, args.prefix, args.experiment)

    logger.info(f"Generated {len(combinations)} parameter combinations")

    if args.dry_run:
        logger.info("Dry run - parameter combinations:")
        for i, params in enumerate(combinations):
            print(f"Task {i}: {params}")
        return

    # Add tasks to queue
    for params in combinations:
        runner.add_task(params)

    # Run experiments
    runner.run()

if __name__ == "__main__":
    main()

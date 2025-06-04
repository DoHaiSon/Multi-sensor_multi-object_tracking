import os, sys, random
import numpy as np
import json
from models.basic import Basic_Model
from models.brg import Brg_Model
from models.brg_rng import Brg_rng_Model
from core.generate_truth import gen_truth, plot_truth
from core.generate_measurements import gen_measurements, plot_measurements
from utils.dataset import save_metadata, save_summary

def set_seed(seed):
    """
    Set the seed for generating random numbers to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use for random number generation
    
    Returns:
        None: Sets seeds for Python's random module and NumPy's random module
    """
    random.seed(seed)  # For Python random
    np.random.seed(seed)  # For NumPy random

def load_configurations(args, writer):
    """
    Load and print the configuration settings, and log them to TensorBoard.

    Args:
        args (Namespace): The arguments containing the configuration settings
        writer (SummaryWriter): The TensorBoard SummaryWriter instance for logging
    
    Returns:
        None: Prints configuration to console and logs to TensorBoard
    """
    # Print the argument summary
    if args.verbose:
        print("\n===== Configurations =====")
        for key, value in vars(args).items():
            if key == 'scenario_params':
                continue
            print(f"{key}: {value}")
    
    # Format the argument summary for TensorBoard with markdown-style table
    arg_summary = "| **Parameter** | **Value** |\n|---|---|\n"
    for key, value in vars(args).items():
        if key == 'scenario_params':
            continue
        arg_summary += f"| {key} | {value} |\n"

    # Log the argument summary to TensorBoard
    writer.add_text('Configurations', arg_summary)

def select_model(args, writer):
    """
    Select the appropriate model based on the configuration settings.

    Args:
        args (Namespace): The arguments containing the configuration settings
        writer (SummaryWriter): The TensorBoard SummaryWriter instance for logging

    Returns:
        model: The selected model instance based on the configuration settings
    
    Raises:
        ValueError: If an invalid model type is specified
    """
    # Convert model name to lowercase for consistency with new config structure
    model_name = args.model.lower()
    
    if model_name in ['basic', 'Basic']:
        model = Basic_Model(args, writer)
    elif model_name in ['brg', 'Brg']:
        model = Brg_Model(args, writer)
    elif model_name in ['brg_rng', 'Brg_rng']:
        model = Brg_rng_Model(args, writer)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    return model

def load_dataset(args, writer):
    """
    Load existing dataset from dataset_dir structure.
    
    Args:
        args (Namespace): Arguments containing dataset configuration. Contains:
            - dataset_dir: Directory containing saved datasets
            - scenario: Scenario name for dataset identification
            - K: Number of time steps
            - enable_logging: Whether to enable visualization logging
        writer (SummaryWriter): TensorBoard writer for logging visualizations
    
    Returns:
        tuple: (truth, measurements, model) where:
            - truth: Ground truth data dictionary
            - measurements: Measurements data dictionary  
            - model: Recreated model instance with sensor configurations
    
    Raises:
        FileNotFoundError: If dataset directory, metadata, or summary files not found
    """
    scenario_name = getattr(args, 'scenario', 'default')
    scenario_dir = os.path.join(args.dataset_dir, f'scenario_{scenario_name}')
    
    # Check if dataset exists
    if not os.path.exists(scenario_dir):
        raise FileNotFoundError(f"Dataset directory not found: {scenario_dir}")
    
    print("\n===== Loading dataset =====")

    # Load metadata
    metadata_file = os.path.join(scenario_dir, 'metadata.json')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load summary
    summary_file = os.path.join(scenario_dir, 'summary.json')
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    K = summary['dataset_summary']['total_timesteps']
    total_tracks = summary['dataset_summary']['total_tracks']
    
    # Initialize truth structure
    truth = {
        'K': K,
        'X': [None] * K,
        'N': np.zeros(K, dtype=int),
        'L': [None] * K,
        'track_list': [None] * K,
        'total_tracks': total_tracks
    }
    
    # Load ground truth timesteps
    truth_dir = os.path.join(scenario_dir, 'ground_truth')
    for k in range(K):
        timestep_file = os.path.join(truth_dir, f'timestep_{k:03d}.npz')
        if os.path.exists(timestep_file):
            data = np.load(timestep_file, allow_pickle=True)
            truth['X'][k] = data['X'] if data['X'].size > 0 else None
            truth['N'][k] = int(data['N'])
            truth['L'][k] = data['L'].tolist() if data['L'].size > 0 else None
            truth['track_list'][k] = data['track_list'].tolist() if data['track_list'].size > 0 else None
    
    # Initialize measurements structure
    num_sensors = metadata['sensors']['num_sensors']
    measurements = {
        'K': K,
        'Z': [[[] for _ in range(num_sensors)] for _ in range(K)],
        'P_D': np.zeros((K, num_sensors)),
        'lambda_c': np.zeros((K, num_sensors))
    }
    
    # Load measurement timesteps
    meas_dir = os.path.join(scenario_dir, 'measurements')
    for k in range(K):
        timestep_file = os.path.join(meas_dir, f'timestep_{k:03d}.npz')
        if os.path.exists(timestep_file):
            data = np.load(timestep_file, allow_pickle=True)
            for s in range(num_sensors):
                sensor_key = f'sensor_{s}_Z'
                if sensor_key in data:
                    measurements['Z'][k][s] = data[sensor_key]
            measurements['P_D'][k, :] = data['P_D']
            measurements['lambda_c'][k, :] = data['lambda_c']
    
    # Recreate the model to get proper sensor configurations
    model = select_model(args, writer)
    
    print(f"Loaded existing dataset from: {scenario_dir}")
    print(f"Dataset contains {total_tracks} tracks and {K} time steps")
    
    # Visualize loaded data if logging is enabled
    if args.enable_logging:
        plot_truth(truth, 0, args.K, writer)
        plot_measurements(args, truth, measurements, model.sensors, 0, args.K, writer)
    
    return truth, measurements, model

def create_dataset(args, writer, seed=None):
    """
    Create a complete dataset including ground truth and measurements.
    
    Args:
        args (Namespace): Arguments containing dataset configuration. Contains:
            - K: Number of time steps
            - save_dataset: Whether to save generated dataset
            - scenario: Scenario name for dataset saving
            - dataset_dir: Directory for saving datasets
            - enable_logging: Whether to enable visualization logging
        writer (SummaryWriter): TensorBoard writer for logging visualizations
        seed (int, optional): Random seed for reproducible dataset generation
    
    Returns:
        tuple: (truth, measurements, model) where:
            - truth: Generated ground truth data dictionary
            - measurements: Generated measurements data dictionary
            - model: Model instance used for generation
    """
    print("\n===== Creating dataset =====")

    # Generate model
    model = select_model(args, writer)

    # Generate ground truth
    truth = gen_truth(args, model.dynamics, seed=seed)
    plot_truth(truth, 0, args.K, writer)

    # Generate measurements
    measurements = gen_measurements(args, model.sensors, truth, seed=seed)
    if args.enable_logging:
        plot_measurements(args, truth, measurements, model.sensors, 0, args.K, writer)

    # Save datasets if enabled (always save as timestep data)
    if hasattr(args, 'save_dataset') and args.save_dataset:
        scenario_name = getattr(args, 'scenario', 'default')
        
        save_metadata(args, model, args.dataset_dir, scenario_name)
        save_summary(truth, measurements, args.dataset_dir, scenario_name)
        
    return truth, measurements, model
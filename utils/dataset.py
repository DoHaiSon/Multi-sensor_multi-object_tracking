import os
import numpy as np
import json
from datetime import datetime

def save_ground_truth(truth, save_dir, scenario_name=None):
    """
    Save ground truth data in the new structure.
    
    Args:
        truth (dict): Ground truth data dictionary. Contains:
            - K: Number of time steps
            - X: List of target states for each time step
            - N: Number of targets at each time step
            - L: Ground truth labels for targets
            - track_list: Absolute index target identities
        save_dir (str): Base directory for saving datasets
        scenario_name (str, optional): Name for the scenario. Defaults to 'default'.
    
    Returns:
        str: Path to the saved ground truth directory
    """
    scenario_name = scenario_name or 'default'
    scenario_dir = os.path.join(save_dir, f'scenario_{scenario_name}')
    truth_dir = os.path.join(scenario_dir, 'ground_truth')
    
    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)
    
    # Save each timestep as separate .npz file
    K = truth['K']
    for k in range(K):
        timestep_data = {}
        
        # Add data for this timestep
        if truth['X'][k] is not None:
            timestep_data['X'] = truth['X'][k]  # Target states
        else:
            timestep_data['X'] = np.array([])
            
        timestep_data['N'] = truth['N'][k]  # Number of targets
        
        if truth['L'][k] is not None:
            # Convert labels list to numpy array for saving
            labels_array = np.array(truth['L'][k], dtype=object)
            timestep_data['L'] = labels_array
        else:
            timestep_data['L'] = np.array([])
            
        if truth['track_list'][k] is not None:
            timestep_data['track_list'] = np.array(truth['track_list'][k])
        else:
            timestep_data['track_list'] = np.array([])
        
        # Save timestep file
        timestep_file = os.path.join(truth_dir, f'timestep_{k:03d}.npz')
        np.savez_compressed(timestep_file, **timestep_data)
    
    print(f"Ground truth saved to: {truth_dir}")
    return truth_dir

def save_measurements(measurements, save_dir, scenario_name=None):
    """
    Save measurements data in the new structure.
    
    Args:
        measurements (dict): Measurements data dictionary. Contains:
            - K: Number of time steps
            - Z: List of measurements for each sensor at each time step
            - P_D: Detection probability for each sensor at each time step
            - lambda_c: Expected number of clutter returns for each sensor
        save_dir (str): Base directory for saving datasets
        scenario_name (str, optional): Name for the scenario. Defaults to 'default'.
    
    Returns:
        str: Path to the saved measurements directory
    """
    scenario_name = scenario_name or 'default'
    scenario_dir = os.path.join(save_dir, f'scenario_{scenario_name}')
    meas_dir = os.path.join(scenario_dir, 'measurements')
    
    if not os.path.exists(meas_dir):
        os.makedirs(meas_dir)
    
    # Save each timestep as separate .npz file
    K = measurements['K']
    num_sensors = len(measurements['Z'][0])
    
    for k in range(K):
        timestep_data = {}
        
        # Save measurements for each sensor
        for s in range(num_sensors):
            if len(measurements['Z'][k][s]) > 0:
                timestep_data[f'sensor_{s}_Z'] = measurements['Z'][k][s]
            else:
                timestep_data[f'sensor_{s}_Z'] = np.array([])
        
        # Save detection probabilities and clutter rates
        timestep_data['P_D'] = measurements['P_D'][k, :]
        timestep_data['lambda_c'] = measurements['lambda_c'][k, :]
        timestep_data['num_sensors'] = num_sensors
        
        # Save timestep file
        timestep_file = os.path.join(meas_dir, f'timestep_{k:03d}.npz')
        np.savez_compressed(timestep_file, **timestep_data)
    
    print(f"Measurements saved to: {meas_dir}")
    return meas_dir

def save_metadata(args, model, save_dir, scenario_name=None):
    """
    Save scenario metadata and model configuration.
    
    Args:
        args (Namespace): Arguments containing configuration parameters
        model (object): Model instance with sensor and parameter configurations
        save_dir (str): Base directory for saving datasets
        scenario_name (str, optional): Name for the scenario. Defaults to 'default'.
    
    Returns:
        str: Path to the saved metadata file
    """
    scenario_name = scenario_name or 'default'
    scenario_dir = os.path.join(save_dir, f'scenario_{scenario_name}')
    
    if not os.path.exists(scenario_dir):
        os.makedirs(scenario_dir)
    
    # Get sensor types - handle both object and dictionary formats
    sensor_types = []
    if hasattr(model, 'sensors'):
        for sensor in model.sensors:
            if hasattr(sensor, 'sensor_type'):
                # New sensor object format
                sensor_types.append(sensor.sensor_type)
            elif isinstance(sensor, dict) and 'type' in sensor:
                # Old dictionary format
                sensor_types.append(sensor['type'])
            else:
                # Fallback
                sensor_types.append('unknown')
    
    # Prepare metadata
    metadata = {
        'scenario_name': scenario_name,
        'created_at': datetime.now().isoformat(),
        'model_type': args.model,
        'time_steps': args.K,
        'seed': getattr(args, 'seed', None),
        'use_seed': getattr(args, 'use_seed', False),
        
        # Scenario parameters
        'scenario_params': {
            'nbirths': args.scenario_params['nbirths'],
            'xstart_shape': args.scenario_params['xstart'].shape,
            'tbirth': args.scenario_params['tbirth'].tolist() if hasattr(args.scenario_params['tbirth'], 'tolist') else args.scenario_params['tbirth'],
            'tdeath': args.scenario_params['tdeath'].tolist() if hasattr(args.scenario_params['tdeath'], 'tolist') else args.scenario_params['tdeath'],
            'wturn': args.scenario_params['wturn']
        },
        
        # Model parameters
        'model_params': {
            'CT_model': args.CT,
            'T': args.T,
            'sigma_vel': args.sigma_vel,
            'sigma_turn': args.sigma_turn,
            'P_S': args.P_S,
            'P_D': args.P_D,
            'lambda_c': args.lambda_c
        },
        
        # Sensor information
        'sensors': {
            'num_sensors': len(model.sensors) if hasattr(model, 'sensors') else 0,
            'sensor_types': sensor_types
        }
    }
    
    metadata_file = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")
    return metadata_file

def save_summary(truth, measurements, save_dir, scenario_name=None):
    """
    Save dataset summary statistics.
    
    Args:
        truth (dict): Ground truth data dictionary for statistics calculation
        measurements (dict): Measurements data dictionary for statistics calculation
        save_dir (str): Base directory for saving datasets
        scenario_name (str, optional): Name for the scenario. Defaults to 'default'.
    
    Returns:
        str: Path to the saved summary file
    """
    scenario_name = scenario_name or 'default'
    scenario_dir = os.path.join(save_dir, f'scenario_{scenario_name}')
    
    if not os.path.exists(scenario_dir):
        os.makedirs(scenario_dir)
    
    # Calculate summary statistics
    total_targets_per_timestep = truth['N']
    max_targets = int(np.max(total_targets_per_timestep))
    avg_targets = float(np.mean(total_targets_per_timestep))
    
    # Calculate measurement statistics
    total_measurements_per_timestep = []
    for k in range(measurements['K']):
        total_meas = 0
        for s in range(len(measurements['Z'][k])):
            if len(measurements['Z'][k][s]) > 0:
                total_meas += measurements['Z'][k][s].shape[1] if measurements['Z'][k][s].ndim > 1 else 1
        total_measurements_per_timestep.append(total_meas)
    
    summary = {
        'dataset_summary': {
            'total_timesteps': truth['K'],
            'total_tracks': truth['total_tracks'],
            'targets_per_timestep': {
                'max': max_targets,
                'min': int(np.min(total_targets_per_timestep)),
                'average': avg_targets,
                'total_observations': int(np.sum(total_targets_per_timestep))
            },
            'measurements_per_timestep': {
                'max': int(np.max(total_measurements_per_timestep)),
                'min': int(np.min(total_measurements_per_timestep)),
                'average': float(np.mean(total_measurements_per_timestep)),
                'total_measurements': int(np.sum(total_measurements_per_timestep))
            }
        },
        'file_structure': {
            'ground_truth_files': truth['K'],
            'measurement_files': measurements['K'],
            'file_format': 'npz_compressed'
        },
        'generated_at': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(scenario_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    return summary_file

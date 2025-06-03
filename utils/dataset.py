import os
import pickle
import json
from datetime import datetime

def save_timestep_data(truth_k, measurements_k, k, save_dir, scenario_name=None):
    """Save ground truth and measurements for a single time step as features and labels."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    prefix = f"{scenario_name}_" if scenario_name else ""
    features_filename = f"{prefix}features_k{k:04d}.pkl"
    labels_filename = f"{prefix}labels_k{k:04d}.pkl"
    
    features_path = os.path.join(save_dir, features_filename)
    labels_path = os.path.join(save_dir, labels_filename)
    
    features = {
        'time_step': k,
        'measurements': measurements_k,
        'num_sensors': len(measurements_k),
        'timestamp': datetime.now().isoformat()
    }
    
    labels = {
        'time_step': k,
        'X': truth_k['X'],
        'N': truth_k['N'],
        'L': truth_k['L'],
        'track_list': truth_k['track_list'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f)
    
    return features_path, labels_path

def save_all_timesteps(truth, measurements, save_dir, scenario_name=None):
    """Save all time steps as individual feature and label files."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    saved_files = []
    K = truth['K']
    
    print(f"Saving {K} time steps to {save_dir}")
    
    for k in range(K):
        truth_k = {
            'X': truth['X'][k],
            'N': truth['N'][k],
            'L': truth['L'][k],
            'track_list': truth['track_list'][k]
        }
        
        measurements_k = measurements['Z'][k]
        
        features_path, labels_path = save_timestep_data(
            truth_k, measurements_k, k, save_dir, scenario_name
        )
        
        saved_files.append((features_path, labels_path))
        
        if (k + 1) % 10 == 0:
            print(f"Saved time steps 0-{k}")
    
    print(f"Completed saving all {K} time steps")
    return saved_files

def create_dataset_manifest(saved_files, save_dir, scenario_name=None):
    """Create a manifest file listing all saved time step files."""
    prefix = f"{scenario_name}_" if scenario_name else ""
    manifest_filename = f"{prefix}dataset_manifest.json"
    manifest_path = os.path.join(save_dir, manifest_filename)
    
    manifest = {
        'scenario': scenario_name,
        'total_timesteps': len(saved_files),
        'created_at': datetime.now().isoformat(),
        'files': []
    }
    
    for k, (features_path, labels_path) in enumerate(saved_files):
        manifest['files'].append({
            'time_step': k,
            'features_file': os.path.basename(features_path),
            'labels_file': os.path.basename(labels_path),
            'features_path': features_path,
            'labels_path': labels_path
        })
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Dataset manifest saved to: {manifest_path}")
    return manifest_path

def save_complete_dataset(truth, measurements, model, args, save_dir, filename=None):
    """Save complete dataset including truth, measurements, model, and args."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_dataset_{timestamp}.pkl"
    
    filepath = os.path.join(save_dir, filename)
    
    dataset = {
        'truth': truth,
        'measurements': measurements,
        'model': model,
        'args': args,
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'K': args.K,
            'total_tracks': truth['total_tracks'],
            'num_sensors': len(model.sensors) if hasattr(model, 'sensors') else 0
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Complete dataset saved to: {filepath}")
    return filepath

def save_dataset_info(dataset_info, save_dir, filename=None):
    """Save dataset information as JSON for easy inspection."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_info_{timestamp}.json"
    
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(dataset_info, f, indent=2, default=str)
    
    print(f"Dataset info saved to: {filepath}")
    return filepath

def load_dataset_file(filepath):
    """Load dataset from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset loaded from: {filepath}")
    return dataset

def load_timestep_data(features_path, labels_path):
    """Load features and labels for a single time step."""
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    return features, labels

def validate_dataset(dataset):
    """Validate loaded dataset structure."""
    required_keys = ['truth', 'measurements', 'model', 'args']
    
    for key in required_keys:
        if key not in dataset:
            raise ValueError(f"Missing required key in dataset: {key}")
    
    print("Dataset validation passed")
    return True

def save_dataset_files(truth, measurements, model, args, seed=None):
    """Save dataset files in various formats."""
    data_dir = os.path.join(args.dataset_dir, 'data')
    complete_dir = os.path.join(args.dataset_dir, 'complete')
    
    scenario_name = getattr(args, 'scenario', 'default')
    saved_files = save_all_timesteps(truth, measurements, data_dir, scenario_name)
    
    manifest_path = create_dataset_manifest(saved_files, data_dir, scenario_name)
    
    dataset_filepath = save_complete_dataset(truth, measurements, model, args, complete_dir)
    
    dataset_info = {
        'scenario': scenario_name,
        'K': args.K,
        'total_tracks': truth['total_tracks'],
        'num_sensors': len(model.sensors),
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'dataset_filepath': dataset_filepath,
        'data_directory': data_dir,
        'total_files': len(saved_files) * 2,
        'manifest_path': manifest_path
    }
    info_path = save_dataset_info(dataset_info, complete_dir)
    
    return {
        'data_dir': data_dir,
        'complete_dir': complete_dir,
        'dataset_filepath': dataset_filepath,
        'manifest_path': manifest_path,
        'info_path': info_path,
        'saved_files': saved_files
    }

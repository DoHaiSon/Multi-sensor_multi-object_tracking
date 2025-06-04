import yaml
import argparse
import os
import datetime
import importlib
from utils.print_utils import setup_print

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary with merged base config if specified
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # If base_config exists, load and merge with current config
    if 'base_config' in config:
        base_file = os.path.join('configs', f"{config['base_config']}.yaml")
        with open(base_file, 'r') as f:
            base_config = yaml.safe_load(f)
        # Merge base config with current config (current config takes precedence)
        base_config.update(config)
        config = base_config
    
    return config

def get_args(override_args=None):
    """
    Parse command line arguments and load configuration from YAML file.
    
    Args:
        override_args (list, optional): List of arguments to override command line args.
            Used primarily for testing. If None, uses sys.argv.
    
    Returns:
        Namespace: Parsed arguments with all configuration parameters
    """
    parser = argparse.ArgumentParser(description='Multi-sensor Multi-object Tracking')
    
    # Model selection
    parser.add_argument('--model', type=str, default='brg_rng', 
                       choices=['brg', 'brg_rng', 'mixed'],
                       help='Model type: brg (bearing only), brg_rng (bearing-range only), mixed (mixed brg and brg_rng)')
    
    # Add argument for config file path
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    
    # Parse command line arguments
    if override_args is not None:
        args = parser.parse_args(override_args)
    else:
        args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Update args with values from config file
    for k, v in config.items():
        setattr(args, k, v)

    # Check verbose mode and setup print
    setup_print(args.verbose)
    
    # Create a default log directory
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.current_time = current_time

    if args.enable_logging:
        if args.log_dir is None:  # Create log directory if not provided
            default_log_dir = os.path.join('logs', f'run_{current_time}')
            os.makedirs(default_log_dir, exist_ok=True)
            args.log_dir = default_log_dir
        else:
            # If log_dir is provided, ensure it doesn't duplicate run names
            log_dir = os.path.join('logs', args.log_dir)
            os.makedirs(log_dir, exist_ok=True)
            args.log_dir = log_dir

    # Load scenario
    try:
        scenario_module = importlib.import_module(f'examples.{args.scenario}')
        args.scenario_params = scenario_module.get_scenario()
    except ImportError:
        raise ValueError(f"Scenario '{args.scenario}' not found in examples folder.")
    
    # Load model-specific config to get num_sensors
    model_config_path = os.path.join('configs', 'sensors', f'{args.model.lower()}.yaml')
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        if 'num_sensors' in model_config:
            args.num_sensors = model_config['num_sensors']
        else:
            # Fallback: try to get from sensor configs length
            if 'sensors' in model_config:
                if 'sensor_configs' in model_config['sensors']:
                    args.num_sensors = len(model_config['sensors']['sensor_configs'])
                elif 'positions' in model_config['sensors']:
                    args.num_sensors = len(model_config['sensors']['positions'])
                else:
                    args.num_sensors = 4  # Default fallback
            else:
                args.num_sensors = 4  # Default fallback
    else:
        args.num_sensors = 4  # Default fallback
    
    # Validate num_sensors
    if not (1 <= args.num_sensors <= 6):
        raise ValueError(f"num_sensors must be between 1 and 6, got {args.num_sensors}")
    
    return args
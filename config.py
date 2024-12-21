import argparse
import os
import datetime
import numpy as np  
import importlib

def str2bool(v):
    """Convert string to boolean."""
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def define_args():
    """Define all arguments"""
    parser = argparse.ArgumentParser(description='Multi-Sensor Multi-Object Tracking with Random Finite Set Theory')

    # Model parameters
    parser.add_argument('--model', default='Basic', type=str,
                        choices=['Basic', 'Brg', 'Brg_rng'], help='Model to use for generating data')
    parser.add_argument('--P_D', default=0.95, type=float, help='Probability of detection')
    parser.add_argument('--lambda_c', default=10, type=float, help='Average number of clutter per frame')
    parser.add_argument('--K', default=100, type=int, help='length of data/number of scans')

    # Dynamic parameters
    parser.add_argument('--x_dim', type=int, default=5, help='Dimension of state vector x (x, v_x, y, v_y, omega)')
    parser.add_argument('--z_dim', type=int, default=2, help='Dimension of measurement vector z (2 for Dynamic, 1 for Brg and Brg_rng)')
    parser.add_argument('--v_dim', type=int, default=3, help='Dimension of process noise vector v')
    parser.add_argument('--T', type=float, default=1.0, help='Sampling time interval')
    parser.add_argument('--sigma_vel', type=float, default=5.0, help='Velocity noise standard deviation')
    parser.add_argument('--sigma_turn', type=float, default=np.pi/180, help='Turn rate noise standard deviation')

    # Survival/death parameters
    parser.add_argument('--P_S', type=float, default=0.95, help='Probability of survival')
    parser.add_argument('--Q_S', type=float, default=1 - 0.95, help='Probability of death')
    parser.add_argument('--P_S_clt', type=float, default=0.9, help='Probability of survival for clutter')

    # Birth parameters
    parser.add_argument('--fixed_birth', default=True, type=str2bool, help='Use pre-defined birth model (True) or random birth model (False)')

    # Observation parameters
    parser.add_argument('--pdf_c', type=float, default=1/(np.pi * 4000), help='Probability density function for clutter')
    parser.add_argument('--range_c_1', type=float, nargs=4, default=[-np.pi/2, np.pi/2, 0, 4000], help='Range clutter 1')
    parser.add_argument('--range_c_2', type=float, nargs=4, default=[np.pi/2, 3*np.pi/2, 0, 4000], help='Range clutter 2')
    parser.add_argument('--D', type=float, nargs=2, default=[2*np.pi/180, 10], help='Diagonal elements for D matrix')
    parser.add_argument('--CT', default=True, type=str2bool, help='Use Constant Turn (True) or Constant Velocity (False) model')

    # Sensor noise parameters
    parser.add_argument('--bearing_D', type=float, default=0.2*np.pi/180, help='Bearing measurement noise standard deviation (radians)')
    parser.add_argument('--range_D', type=float, default=10.0, help='Range measurement noise standard deviation')

    # Generate truth parameters
    parser.add_argument('--scenario', type=str, default='scenario1', help='Scenario file name in /examples folder (default: scenario1)')

    # Running parameters
    parser.add_argument('--verbose', default=True, type=str2bool, help='Enable verbose output')
    parser.add_argument('--enable_logging', type=bool, default=True, help='Enable/disable logging and log directory creation')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs (auto-generated if not provided)')

    return parser

def get_args(args_list=None):
    """
    Get arguments from command line or from a list

    Parameters:
    -----------
    args_list : list, optional
        List of arguments to parse. If None, uses sys.argv[1:]
        For testing, pass an empty list [] to get default values

    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments
    """
    parser = define_args()

    # Parse the arguments
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    # Create a default log directory
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.current_time = current_time

    if args.enable_logging:
        if args.log_dir is None:  # Create log directory if not provided
            default_log_dir = os.path.join('logs', f'run_{current_time}')
            os.makedirs(default_log_dir, exist_ok=True)
            args.log_dir = default_log_dir  # Assign default log directory to args.log_dir
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
    
    # Adjust z_dim based on model type
    if args.model == 'Brg' or args.model == 'Brg_rng':
        args.z_dim = 1

    return args
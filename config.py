import argparse
import os
import datetime
import numpy as np  

def str2bool(v):
    """Convert string to boolean."""
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Multi-Sensor Multi-Object Tracking with Random Finite Set Theory')

    # Model parameters
    parser.add_argument('--model', default='Dynamic', type=str,
                        choices=['Dynamic', 'Brg', 'Brg_rng'], help='Model to use for generating data')
    parser.add_argument('--P_D', default=0.95, type=float, help='Probability of detection')
    parser.add_argument('--lambda_c', default=10, type=float, help='Average number of clutter per frame')
    parser.add_argument('--P_S', type=float, default=0.95, help='Probability of survival')
    parser.add_argument('--Q_S', type=float, default=1 - 0.95, help='Probability of death')
    parser.add_argument('--P_S_clt', type=float, default=0.9, help='Probability of survival for clutter')

    # Dynamic model parameters
    parser.add_argument('--x_dim', type=int, default=5, help='Dimension of state vector x (x, v_x, y, v_y, omega)')
    parser.add_argument('--z_dim', type=int, default=2, help='Dimension of measurement vector z (azimuth, range)')
    parser.add_argument('--v_dim', type=int, default=3, help='Dimension of process noise vector v')
    parser.add_argument('--T', type=float, default=1.0, help='Sampling time interval')
    parser.add_argument('--sigma_vel', type=float, default=5.0, help='Velocity noise standard deviation')
    parser.add_argument('--sigma_turn', type=float, default=np.pi/180, help='Turn rate noise standard deviation')

    # Running parameters
    parser.add_argument('--verbose', default=True, type=str2bool, help='Enable verbose output')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs (auto-generated if not provided)')

    # Parse the arguments
    args = parser.parse_args()

    # Create a default log directory
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.current_time = current_time

    if args.log_dir is None:  # Create log directory if not provided
        default_log_dir = os.path.join('logs', f'run_{current_time}')
        os.makedirs(default_log_dir, exist_ok=True)
        args.log_dir = default_log_dir  # Assign default log directory to args.log_dir
    else:
        # If log_dir is provided, ensure it doesn't duplicate run names
        log_dir = os.path.join('logs', args.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        args.log_dir = log_dir

    return args
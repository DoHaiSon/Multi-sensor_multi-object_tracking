import os, sys
import numpy as np

from config import get_args 
from utils.logger import Logger
from core.generate_truth import gen_truth, plot_truth
from core.generate_measurements import gen_measurements, plot_measurements
from utils.common import set_seed, load_configurations, select_model

if __name__ == '__main__':
    args = get_args()

    seed = args.seed if args.use_seed else None
    
    # Set global seed if needed
    if args.use_seed:
        set_seed(args.seed)

    # Configure the logger
    writer = Logger(enable_logging=args.enable_logging, log_dir=args.log_dir)

    # Load the configurations
    load_configurations(args, writer)

    # Generate model
    model = select_model(args, writer)

    # Generate ground truth
    truth = gen_truth(args, model.dynamics, seed=seed)

    # Visualize ground truth
    plot_truth(truth, 0, args.K, writer)

    # Generate measurements
    measurements = gen_measurements(args, model.sensors, truth, seed=seed)

    # Visualize measurements: heavy computation (RAM), consider commenting out
    plot_measurements(args, truth, measurements, model.sensors, 0, args.K, writer)

    writer.close()
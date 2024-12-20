import os, sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import get_args 
from utils.common import set_seed, load_configurations, select_model

if __name__ == '__main__':
    args = get_args()
    set_seed(2808)  # Set seed for reproducibility

    # Configure the logger
    writer = SummaryWriter(log_dir=args.log_dir)

    # Load the configurations
    load_configurations(args, writer)

    # Generate model: 
    model = select_model(args, writer)

    writer.close()
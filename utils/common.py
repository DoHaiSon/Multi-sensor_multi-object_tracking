import os, sys, random
import numpy as np
from models.basic import Basic_Model
from models.brg import Brg_Model
from models.brg_rng import Brg_rng_Model

def set_seed(seed):
    """
    Set the seed for generating random numbers to ensure reproducibility.
    
    Parameters:
    seed (int): The seed value to use for random number generation.
    
    This function sets the seed for:
    - Python's built-in random module
    - NumPy's random module
    """
    random.seed(seed)  # For Python random
    np.random.seed(seed)  # For NumPy random

def load_configurations(args, writer):
    """
    Load and print the configuration settings, and log them to TensorBoard.

    Parameters:
    args (Namespace): The arguments containing the configuration settings.
    writer (SummaryWriter): The TensorBoard SummaryWriter instance for logging.
    
    This function performs the following tasks:
    - Prints the configuration settings to the console.
    - Formats the configuration settings as a markdown-style table.
    - Logs the formatted configuration settings to TensorBoard.
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

    Parameters:
    args (Namespace): The arguments containing the configuration settings.
    writer (SummaryWriter): The TensorBoard SummaryWriter instance for logging.

    Returns:
    model: The selected model based on the configuration settings.
    """
    if args.model == 'Basic':
        model = Basic_Model(args, writer)
    elif args.model == 'Brg':
        model = Brg_Model(args, writer)
    elif args.model == 'Brg_rng':
        model = Brg_rng_Model(args, writer)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    return model
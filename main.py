import os, sys
from config import get_args 
from utils.logger import Logger
from utils.common import set_seed, create_dataset, load_dataset, load_configurations

def main():
    """Main function to run the multi-sensor multi-object tracking simulation."""
    args = get_args()

    seed = args.seed if args.use_seed else None
    
    # Set global seed if needed
    if args.use_seed:
        set_seed(args.seed)

    # Configure the logger
    writer = Logger(enable_logging=args.enable_logging, log_dir=args.log_dir)

    # Load configurations
    load_configurations(args, writer)

    # Load dataset if specified, otherwise create new one
    if args.load_dataset:
        truth, measurements, model = load_dataset(args, writer)
    else:
        truth, measurements, model = create_dataset(args, writer, seed)

    writer.close()
    print("Completed!")

if __name__ == '__main__':
    main()
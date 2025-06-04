# Multi-Sensor Multi-Object Tracking with Random Finite Set Theory

This repository provides an implementation of a **multi-sensor, multi-object tracking (MS-MOT)** system based on **Random Finite Set (RFS) theory** in Python.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Viewing Results](#viewing-results)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/DoHaiSon/Multi-sensor_multi-object_tracking.git
cd Multi-sensor_multi-object_tracking

# Create conda environment from environment file
conda env create -f env.yml
conda activate MOT

# Run a basic example
python main.py 
```

## Project Structure

```
Multi-sensor_multi-object_tracking/
├── configs/          # Configuration files
│   ├── sensors/      # Sensor-specific configs
│   └── default.yaml  # Default configuration
├── core/             # Core tracking algorithms
├── docs/             # Detailed documentation
├── examples/         # Example scenarios
├── models/           # System model implementations
├── sensors/          # Sensors
├── tests/            # Test suite
├── utils/            # Utility functions
├── main.py           # Main entry point
└── env.yml           # Conda environment file
```

## Documentation

Detailed documentation is organized into separate guides:

- **[Sensor Configuration](docs/sensors.md)** - Setting up various sensor types

## Viewing Results

Monitor results using TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir logs

# Or view latest log automatically
python utils/show_log.py
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details on:

- Code style and standards
- Submitting pull requests
- Reporting issues
- Development workflow

## License

This project is licensed under the **GPL-3.0 License** - see the [LICENSE](LICENSE) file for details.

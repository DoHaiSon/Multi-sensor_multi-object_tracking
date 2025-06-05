import numpy as np
import yaml
import os
from utils.helpers import log_params, plot_sensor_positions
from sensors.sensor_factory import SensorFactory

class Brg_rng_Model:
    def __init__(self, args, writer):
        """
        Initialize basic bearing-range sensor model.
        
        Args:
            args: Arguments from parse_args containing model configuration
            writer: TensorboardX SummaryWriter for logging parameters
        
        Returns:
            None: Initializes all model components
        """
        self.args = args
        self.writer = writer

        # Load sensor configuration from YAML
        self.sensor_config = self.load_sensor_config()

        # Initialize dynamics
        self.dynamics = {
            'x_dim': self.args.x_dim,  # x, vx, y, vy, omega
            'v_dim': self.args.v_dim,
            'T': self.args.T,
            'sigma_vel': self.args.sigma_vel,
            'sigma_turn': self.args.sigma_turn,
            'bt': None,
            'B2': None,
            'B': None,
            'Q': None,
            'pdvarfac_tg': 0.01 ** 2
        }
        # Calculate noise-related variables
        self.calculate_noise_matrices()

        # Survival/death model
        self.survival = {
            'P_S': self.args.P_S,
            'Q_S': self.args.Q_S,
            'P_S_clt': self.args.P_S_clt
        }

        # Birth model
        self.birth = self.initialize_birth_model()

        # Multisensor observation model
        self.sensors = self.initialize_sensors()

        # Log dynamics to TensorBoard
        log_params(args, self, self.writer)

        # Visualize sensor positions
        plot_sensor_positions(self, self.writer)

    def _evaluate_expressions(self, config):
        """
        Evaluate string expressions containing numpy functions in config.
        
        Args:
            config (dict): Configuration dictionary that may contain string expressions
            
        Returns:
            dict: Configuration with evaluated expressions
        """
        import copy
        
        def evaluate_recursive(obj):
            if isinstance(obj, dict):
                return {k: evaluate_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [evaluate_recursive(item) for item in obj]
            elif isinstance(obj, str) and 'np.' in obj:
                # Safely evaluate numpy expressions
                try:
                    return eval(obj, {"np": np})
                except:
                    return obj  # Return original string if evaluation fails
            else:
                return obj
        
        return evaluate_recursive(copy.deepcopy(config))

    def load_sensor_config(self):
        """
        Load sensor configuration from YAML file.
        
        Args:
            None
        
        Returns:
            dict: Sensor configuration dictionary for bearing-range sensors
        """
        config_path = os.path.join('configs', 'sensors', 'brg_rng.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return self._evaluate_expressions(config)
        else:
            return self.get_default_config()

    def get_default_config(self):
        """
        Get default configuration if YAML file is not found.
        
        Args:
            None
        
        Returns:
            dict: Default configuration with bearing-range sensor parameters
        """
        return {
            'num_sensors': 4,
            'sensors': {
                'positions': [[-2000, 0], [2000, 0], [2000, 2000], [-2000, 2000]],
                'velocities': [[0, 0], [0, 0], [0, -10], [10, 0]],
                'bearing_range': {
                    'type': 'brg_rng',
                    'z_dim': 2,
                    'noise_std': [2*np.pi/180, 10],  # D = diag([2*pi/180, 10])
                    'detection_prob': [0.95, 0.95],  # [P_D P_D]
                    'clutter_rate': [10, 10],  # [lambda_c lambda_c]
                    'pdf_c': 1/(np.pi * 4000)  # 1/(pi * 4000)
                },
                'clutter_ranges': [
                    [[-np.pi/2, np.pi/2], [0, 4000]],  # [-pi/2, pi/2; 0, 4000]
                    [[np.pi/2, 3*np.pi/2], [0, 4000]],   # [pi/2, 3*pi/2; 0, 4000]
                    [[np.pi/2, 3*np.pi/2], [0, 4000]],   # [pi/2, 3*pi/2; 0, 4000]
                    [[-np.pi/2, np.pi/2], [0, 4000]]   # [-pi/2, pi/2; 0, 4000]
                ]
            },
            'birth_positions': [
                [-1500, 0, 250, 0, 0],
                [-250, 0, 1000, 0, 0],
                [250, 0, 750, 0, 0],
                [1000, 0, 1500, 0, 0]
            ],
            'birth_covariance_diag': [50, 50, 50, 50, 6*np.pi/180]  # 6*pi/180
        }

    def calculate_noise_matrices(self):
        """
        Calculate noise-related variables in the dynamics dictionary.
        
        Args:
            None
        
        Returns:
            None: Updates self.dynamics with computed matrices bt, B2, B, Q
        """
        # Calculate bt (2x1 vector)
        self.dynamics['bt'] = self.dynamics['sigma_vel'] * np.array([(self.dynamics['T'] ** 2) / 2, self.dynamics['T']])

        # Calculate B2 (5x3 matrix)
        self.dynamics['B2'] = np.array([
            [self.dynamics['bt'][0], 0.0, 0.0],
            [self.dynamics['bt'][1], 0.0, 0.0],
            [0.0, self.dynamics['bt'][0], 0.0],
            [0.0, self.dynamics['bt'][1], 0.0],
            [0.0, 0.0, self.dynamics['T'] * self.dynamics['sigma_turn']]
        ])

        # Calculate B (identity matrix with size v_dim)
        self.dynamics['B'] = np.eye(self.dynamics['v_dim'])
        
        # Calculate Q
        self.dynamics['Q'] = np.dot(self.dynamics['B'], self.dynamics['B'].T)

    def initialize_birth_model(self):
        """
        Initialize birth model parameters from args configuration.
        
        Args:
            None
        
        Returns:
            list: List of birth model dictionaries for target birth process
        """
        birth = []
        
        # Load birth configuration from args
        if hasattr(self.args, 'birth_model'):
            birth_config = self.args.birth_model
            positions = birth_config.get('birth_positions')
            diag_values = birth_config.get('birth_covariance_diag')
            birth_prob = birth_config.get('birth_probability', 0.01)
        else:
            raise ValueError("Birth model configuration not found in args. Please check default.yaml configuration.")
        
        # Handle string expressions in diag_values
        evaluated_diag = []
        for val in diag_values:
            if isinstance(val, str) and 'np.' in val:
                try:
                    evaluated_diag.append(eval(val, {"np": np}))
                except:
                    evaluated_diag.append(val)
            else:
                evaluated_diag.append(val)
        
        B_diag = np.diag(evaluated_diag)
        P = np.dot(B_diag, B_diag)
        
        for position in positions:
            birth_model = {
                'L': 1,
                'r': birth_prob,
                'w': np.array([1.0]),
                'm': np.array(position).reshape(-1, 1),
                'B': B_diag,
                'P': P
            }
            birth.append(birth_model)
        
        return birth

    def initialize_sensors(self):
        """
        Initialize sensor parameters from config.
        
        Args:
            None
        
        Returns:
            list: List of bearing-range sensor objects with velocity parameters
        """
        sensors = []
        
        config = self.sensor_config['sensors']
        positions = config['positions']
        velocities = config['velocities']
        bearing_range_config = config['bearing_range']
        clutter_ranges = config['clutter_ranges']
        
        num_sensors = self.sensor_config.get('num_sensors', len(positions))
        
        for i in range(num_sensors):
            pos = positions[i]
            vel = velocities[i]
            clutter_range = clutter_ranges[i]
            
            # Create sensor configuration for SensorFactory
            sensor_config = {
                'type': bearing_range_config['type'],
                'id': i,
                'position': pos,
                'P_D_rng': bearing_range_config['detection_prob'],
                'lambda_c_rng': bearing_range_config['clutter_rate'],
                'R': np.diag([std**2 for std in bearing_range_config['noise_std']]),
                'range_c': clutter_range
            }
            
            # Create sensor using SensorFactory
            sensor_obj = SensorFactory.create_sensor(sensor_config)
            # Add velocity if needed for compatibility
            if hasattr(sensor_obj, 'velocity'):
                sensor_obj.velocity = np.array(vel)
            sensors.append(sensor_obj)

        return sensors
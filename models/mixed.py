import numpy as np
import yaml
import os
from utils.helpers import log_params, plot_sensor_positions
from sensors.sensor_factory import SensorFactory


class Mixed_Model:
    def __init__(self, args, writer):
        """
        Initialize mixed bearing and bearing-range sensor model.
        
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
            'z_dim': self.args.z_dim,  # azimuth, range
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

    def load_sensor_config(self):
        """
        Load sensor configuration from YAML file.
        
        Args:
            None
        
        Returns:
            dict: Sensor configuration dictionary with mixed sensor types
        """
        config_path = os.path.join('configs', 'sensors', 'mixed.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self):
        """
        Get default configuration if YAML file is not found.
        
        Args:
            None
        
        Returns:
            dict: Default configuration with bearing and bearing-range sensors
        """
        return {
            'num_sensors': 4,
            'sensors': {
                'sensor_configs': [
                    {
                        'type': 'brg',
                        'position': [-2000, 0],
                        'z_dim': 1,
                        'noise_std': 0.1323,
                        'detection_prob': [0.98, 0.98],
                        'clutter_rate': [10, 15],
                        'clutter_range': [0, 6.283185307179586]
                    },
                    {
                        'type': 'brg',
                        'position': [2000, 0],
                        'z_dim': 1,
                        'noise_std': 0.1323,
                        'detection_prob': [0.98, 0.98],
                        'clutter_rate': [10, 15],
                        'clutter_range': [0, 6.283185307179586]
                    },
                    {
                        'type': 'brg_rng',
                        'position': [2000, 2000],
                        'velocity': [0, -10],
                        'z_dim': 2,
                        'noise_std': [0.0349, 100],
                        'detection_prob': [0.98, 0.98],
                        'clutter_rate': [10, 15],
                        'clutter_range': [[1.5708, 4.7124], [0, 4000]],
                        'pdf_c': 7.957747e-06
                    },
                    {
                        'type': 'brg_rng',
                        'position': [-2000, 2000],
                        'velocity': [10, 0],
                        'z_dim': 2,
                        'noise_std': [0.0349, 100],
                        'detection_prob': [0.98, 0.98],
                        'clutter_rate': [10, 15],
                        'clutter_range': [[-1.5708, 1.5708], [0, 4000]],
                        'pdf_c': 7.957747e-06
                    }
                ]
            }
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
        Initialize birth model parameters from config.
        
        Args:
            None
        
        Returns:
            list: List of birth model dictionaries for target initialization
        """
        birth = []
        positions = self.sensor_config.get('birth_positions', [
            [-1500, 0, 250, 0, 0],
            [-250, 0, 1000, 0, 0],
            [250, 0, 750, 0, 0],
            [1000, 0, 1500, 0, 0]
        ])
        
        diag_values = self.sensor_config.get('birth_covariance_diag', [50, 50, 50, 50, 0.1047])
        B_diag = np.diag(diag_values)
        P = np.dot(B_diag, B_diag)
        
        for position in positions:
            birth_model = {
                'L': 1,
                'r': 0.01,
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
            list: List of mixed sensor objects (bearing and bearing-range)
        """
        sensors = []
        
        sensor_configs = self.sensor_config['sensors']['sensor_configs']
        
        for i, config in enumerate(sensor_configs):
            # Prepare sensor configuration for SensorFactory
            if config['type'] == 'brg':
                R = config['noise_std'] ** 2
            else:  # brg_rng
                R = np.diag([std**2 for std in config['noise_std']])
            
            sensor_config = {
                'type': config['type'],
                'id': i,
                'position': config['position'],
                'P_D_rng': config['detection_prob'],
                'lambda_c_rng': config['clutter_rate'],
                'R': R,
                'range_c': config['clutter_range']
            }
            
            # Create sensor using SensorFactory
            sensor_obj = SensorFactory.create_sensor(sensor_config)
            
            # Add velocity if present
            if 'velocity' in config:
                if hasattr(sensor_obj, 'velocity'):
                    sensor_obj.velocity = np.array(config['velocity'])
                    
            sensors.append(sensor_obj)

        return sensors
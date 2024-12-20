import numpy as np
from utils.helpers import log_params, plot_sensor_positions


class Brg_rng_Model:
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer

        # Initialize dynamics
        self.dynamics = {
            'x_dim': self.args.x_dim,  # x, vx, y, vy, omega
            'z_dim': self.args.z_dim,  # azimuth only
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
        self.sensors = self.initialize_sensors([self.args.P_D, self.args.P_D], [self.args.lambda_c, self.args.lambda_c])

        # Log parameters
        log_params(args, self, self.writer)
        plot_sensor_positions(self, self.writer)

    def calculate_noise_matrices(self):
        """
        Calculate noise-related variables in the dynamics dictionary.
        """
        # Calculate bt (2x1 vector)
        self.dynamics['bt'] = self.dynamics['sigma_vel'] * np.array([(self.dynamics['T'] ** 2) / 2, self.dynamics['T']])

        # Calculate B2 (5x3 matrix) correctly matching MATLAB structure
        self.dynamics['B2'] = np.array([
            [self.dynamics['bt'][0], 0.0, 0.0],             # First row
            [self.dynamics['bt'][1], 0.0, 0.0],             # Second row
            [0.0, self.dynamics['bt'][0], 0.0],             # Third row
            [0.0, self.dynamics['bt'][1], 0.0],             # Fourth row
            [0.0, 0.0, self.dynamics['T'] * self.dynamics['sigma_turn']]  # Fifth row
        ])

        # Calculate B (identity matrix with size v_dim)
        self.dynamics['B'] = np.eye(self.dynamics['v_dim'])
        
        # Calculate Q
        self.dynamics['Q'] = np.dot(self.dynamics['B'], self.dynamics['B'].T)

    def initialize_birth_model(self):
        """Initialize birth model parameters"""
        birth_positions = [
            [-1500, 0, 250, 0, 0],
            [-250, 0, 1000, 0, 0],
            [250, 0, 750, 0, 0],
            [1000, 0, 1500, 0, 0]
        ]
        
        birth = []
        B_diag = np.diag([50, 50, 50, 50, 6*(np.pi/180)])
        P = B_diag @ B_diag.T
        
        for pos in birth_positions:
            birth_model = {
                'L': 1,
                'r': 0.01,
                'w': np.array([1.0]),
                'm': np.array(pos).reshape(-1, 1),
                'B': B_diag,
                'P': P
            }
            birth.append(birth_model)
            
        return birth

    def initialize_sensors(self, detect_prob, clutter_rate):
        """Initialize sensor parameters"""
        sensors = []
        
        # Common parameters for bearing sensors
        brg_pdf_c = 1/(2*np.pi)
        brg_range_c = np.array([0, 2*np.pi])
        brg_D = self.args.bearing_D
        brg_R = brg_D*brg_D
        brg_z_dim = 1
        brg_w_dim = 1

        # Parameters for range-bearing sensors
        range_c_1 = np.array([[-np.pi/2, np.pi/2], [0, 4000]])
        range_c_2 = np.array([[np.pi/2, 3*np.pi/2], [0, 4000]])
        rng_brg_D = np.diag([2*np.pi/180, self.args.range_D])  # Use range_D from args
        rng_brg_R = rng_brg_D @ rng_brg_D.T
        rng_brg_pdf_c = 1/(np.pi * 4000)

        # Sensor configurations
        sensor_configs = [
            # Bearing sensors
            {
                'type': 'brg', 'pos': [-2000, 0], 'vel': None,
                'z_dim': brg_z_dim, 'w_dim': brg_w_dim,
                'D': brg_D, 'R': brg_R,
                'range_c': brg_range_c, 'pdf_c': brg_pdf_c
            },
            {
                'type': 'brg', 'pos': [2000, 0], 'vel': None,
                'z_dim': brg_z_dim, 'w_dim': brg_w_dim,
                'D': brg_D, 'R': brg_R,
                'range_c': brg_range_c, 'pdf_c': brg_pdf_c
            },
            {
                'type': 'brg', 'pos': [2000, 2000], 'vel': [0, -10],
                'z_dim': brg_z_dim, 'w_dim': brg_w_dim,
                'D': brg_D, 'R': brg_R,
                'range_c': brg_range_c, 'pdf_c': brg_pdf_c
            },
            {
                'type': 'brg', 'pos': [-2000, 2000], 'vel': [10, 0],
                'z_dim': brg_z_dim, 'w_dim': brg_w_dim,
                'D': brg_D, 'R': brg_R,
                'range_c': brg_range_c, 'pdf_c': brg_pdf_c
            },
            # Range-Bearing sensors
            {
                'type': 'brg_rng', 'pos': [2000, 2000], 'vel': [0, -10],
                'z_dim': 2, 'w_dim': 2,
                'D': rng_brg_D, 'R': rng_brg_R,
                'range_c': range_c_2, 'pdf_c': rng_brg_pdf_c
            },
            {
                'type': 'brg_rng', 'pos': [-2000, 2000], 'vel': [10, 0],
                'z_dim': 2, 'w_dim': 2,
                'D': rng_brg_D, 'R': rng_brg_R,
                'range_c': range_c_1, 'pdf_c': rng_brg_pdf_c
            }
        ]

        for config in sensor_configs:
            sensor = {
                'type': config['type'],
                'z_dim': config['z_dim'],
                'w_dim': config['w_dim'],
                'X': np.array(config['pos']),
                'D': config['D'],
                'R': config['R'],
                'P_D_rng': detect_prob,
                'P_D': np.mean(detect_prob),
                'Q_D': 1 - np.mean(detect_prob),
                'lambda_c_rng': clutter_rate,
                'lambda_c': np.mean(clutter_rate),
                'range_c': config['range_c'],
                'pdf_c': config['pdf_c']
            }
            
            if config['vel'] is not None:
                sensor['v'] = np.array(config['vel'])
                
            sensors.append(sensor)

        return sensors
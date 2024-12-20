import numpy as np
from utils.helpers import log_params, plot_sensor_positions

class DynamicModel:
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer

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
        self.sensors = self.initialize_sensors([args.P_D, args.P_D], [args.lambda_c, args.lambda_c])

        # Log dynamics to TensorBoard
        log_params(args, self, self.writer)

        # Visualize sensor positions
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
        """
        Initialize birth model parameters.
        """
        birth = []
        positions = [
            [-1500, 0, 250, 0, 0],
            [-250, 0, 1000, 0, 0],
            [250, 0, 750, 0, 0],
            [1000, 0, 1500, 0, 0]
        ]
        B_diag = np.diag([50, 50, 50, 50, 6*(np.pi/180)])
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

    def initialize_sensors(self, detect_prob, clutter_rate):
        """
        Initialize sensor parameters.
        """
        sensors = []
        idx = 0
        pdf_c = self.args.pdf_c
        range_c_1 = np.array(self.args.range_c_1).reshape(2, 2)
        range_c_2 = np.array(self.args.range_c_2).reshape(2, 2)
        D = np.diag(self.args.D)
        R = np.dot(D, D)
        w_dim = self.args.z_dim

        positions = [
            [-2000, 0],
            [2000, 0],
            [2000, 2000],
            [-2000, 2000]
        ]
        velocities = [
            [0, 0],
            [0, 0],
            [0, -10],
            [10, 0]
        ]
        ranges = [range_c_1, range_c_2, range_c_2, range_c_1]
        
        for pos, vel, rng in zip(positions, velocities, ranges):
            sensor = {
                'type': 'brg_rng',
                'z_dim': self.args.z_dim,
                'w_dim': w_dim,
                'X': np.array(pos),
                'v': np.array(vel),
                'D': D,
                'R': R,
                'P_D_rng': detect_prob,
                'P_D': np.mean(detect_prob),
                'Q_D': 1 - np.mean(detect_prob),
                'lambda_c_rng': clutter_rate,
                'lambda_c': np.mean(clutter_rate),
                'range_c': rng,
                'pdf_c': pdf_c
            }
            sensors.append(sensor)
            idx += 1
        
        return sensors
import numpy as np
from utils.helpers import log_params, plot_sensor_positions

class Brg_Model:
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
        self.sensors = self.initialize_sensors([self.args.P_D, self.args.P_D], [self.args.lambda_c, self.args.lambda_c])

        # Log dynamics to TensorBoard
        log_params(args, self, self.writer)

        # Visualize sensor positions
        plot_sensor_positions(self, self.writer)

    def calculate_noise_matrices(self):
        """Calculate noise-related variables in the dynamics dictionary."""
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
        """Initialize birth model parameters."""
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
        """Initialize sensor parameters."""
        sensors = []
        pdf_c = 1/(2*np.pi)
        range_c = np.array([0, 2*np.pi])
        D = self.args.bearing_D
        R = D*D
        w_dim = self.args.z_dim

        # Sensor positions
        positions = [
            [-2000, 0],    # bottom left
            [2000, 0],     # bottom right
            [2000, 2000],  # top right
            [-2000, 2000], # top left
            [0, 2000],     # top middle
            [0, 0]         # bottom middle
        ]

        for pos in positions:
            sensor = {
                'type': 'brg',
                'z_dim': self.args.z_dim,
                'w_dim': w_dim,
                'X': np.array(pos),
                'D': D,
                'R': R,
                'P_D_rng': detect_prob,
                'P_D': np.mean(detect_prob),
                'Q_D': 1 - np.mean(detect_prob),
                'lambda_c_rng': clutter_rate,
                'lambda_c': np.mean(clutter_rate),
                'range_c': range_c,
                'pdf_c': pdf_c
            }
            sensors.append(sensor)

        return sensors
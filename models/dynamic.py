import numpy as np

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

        # Log dynamics to TensorBoard
        self.log_dynamics_table()

    def calculate_noise_matrices(self):
        """
        Calculate noise-related variables in the dynamics dictionary.
        """
        self.dynamics['bt'] = self.dynamics['sigma_vel'] * np.array([(self.dynamics['T'] ** 2) / 2, self.dynamics['T']])
        self.dynamics['B2'] = np.array([
            [self.dynamics['bt'][0], 0, 0, 0],
            [0, self.dynamics['bt'][1], 0, 0],
            [0, 0, self.dynamics['bt'][0], 0],
            [0, 0, 0, self.dynamics['bt'][1]],
            [0, 0, 0, self.dynamics['T'] * self.dynamics['sigma_turn']]
        ])
        self.dynamics['B'] = np.eye(self.dynamics['v_dim'])
        self.dynamics['Q'] = np.dot(self.dynamics['B'], self.dynamics['B'].T)

    def format_array(self, array):
        """
        Format a numpy array as a string without new lines.
        """
        return np.array2string(array, separator=', ', formatter={'all':lambda x: str(x)}).replace('\n', '')

    def log_dynamics_table(self):
        """
        Log the dynamics parameters as a markdown-style table to TensorBoard.
        """
        arg_summary = "| **Parameter** | **Value** |\n|---|---|\n"
        for key, value in self.dynamics.items():
            if isinstance(value, np.ndarray):
                value_str = self.format_array(value)
            else:
                value_str = str(value)
            arg_summary += f"| {key} | {value_str} |\n"
        
        self.writer.add_text('Dynamic model parameters', arg_summary)
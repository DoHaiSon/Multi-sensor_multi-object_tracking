import numpy as np
from utils.helpers import one_line_array
import matplotlib.pyplot as plt

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
        self.log()

        # Visualize sensor positions
        self.plot_sensor_positions()

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

    def log(self):
        """
        Log the dynamic model parameters as a markdown-style table to TensorBoard.
        """
        # Log dynamic parameters
        arg_summary = "| **Parameter** | **Value** |\n|---|---|\n"
        for key, value in self.dynamics.items():
            if isinstance(value, np.ndarray):
                value_str = one_line_array(value)
            else:
                value_str = str(value)
            arg_summary += f"| {key} | {value_str} |\n"
        
        self.writer.add_text('Dynamic parameters', arg_summary)

        # Log survival parameters
        arg_summary = "| **Parameter** | **Value** |\n|---|---|\n"
        for key, value in self.survival.items():
            if isinstance(value, np.ndarray):
                value_str = one_line_array(value)
            else:
                value_str = str(value)
            arg_summary += f"| {key} | {value_str} |\n"
        
        self.writer.add_text('Survival/death parameters', arg_summary)

        # Log birth model parameters
        if not self.args.fixed_birth:
            arg_summary = "| **Birth Model Parameter** | **Value** |\n|---|---|\n"
            for i, birth in enumerate(self.birth):
                arg_summary += f"| **Birth model {i+1}** | |\n"
                for key, value in birth.items():
                    if isinstance(value, np.ndarray):
                        value_str = one_line_array(value)
                    else:
                        value_str = str(value)
                    arg_summary += f"| {key} | {value_str} |\n"
            
            self.writer.add_text('Birth model parameters', arg_summary)

        # Log sensor parameters
        arg_summary = "| **Sensor Parameter** | **Value** |\n|---|---|\n"
        for i, sensor in enumerate(self.sensors):
            arg_summary += f"| **Sensor {i+1}** | |\n"
            for key, value in sensor.items():
                if isinstance(value, np.ndarray):
                    value_str = one_line_array(value)
                else:
                    value_str = str(value)
                arg_summary += f"| {key} | {value_str} |\n"
        
        self.writer.add_text('Sensor parameters', arg_summary)

    def plot_sensor_positions(self):
        """
        Plot the sensor positions and ranges and log the plot to TensorBoard.
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # Increase figure size and DPI for higher resolution
        ax.set_title('Sensor Positions and Ranges')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Different colors for different sensors
        colors = ['r', 'g', 'b', 'm']

        for i, sensor in enumerate(self.sensors):
            pos = sensor['X']
            range_c = sensor['range_c']

            # Plot sensor position
            ax.plot(pos[0], pos[1], 'o', color=colors[i % len(colors)])  # colored dot for sensor position
            ax.text(pos[0], pos[1], f'Sensor {i+1}', fontsize=9, ha='right')

            # Plot sensor range with increased spacing
            theta = np.linspace(range_c[0, 0], range_c[0, 1], 50)
            r = np.linspace(range_c[1, 0], range_c[1, 1], 50)
            theta_grid, r_grid = np.meshgrid(theta, r)
            x_grid = r_grid * np.cos(theta_grid) + pos[0]
            y_grid = r_grid * np.sin(theta_grid) + pos[1]
            ax.plot(x_grid, y_grid, color=colors[i % len(colors)], alpha=0.7)

        ax.grid(True)
        plt.axis('equal')

        # Log the plot to TensorBoard
        self.writer.add_figure('Sensor Positions and Ranges', fig)

        # Close the plot to free memory
        plt.close(fig)
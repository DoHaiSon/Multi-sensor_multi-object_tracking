import numpy as np
import matplotlib.pyplot as plt

def one_line_array(array):
    """
    Format a numpy array as a string without new lines.
    """
    return np.array2string(array, separator=', ', formatter={'all':lambda x: str(x)}).replace('\n', '')


def log_params(args, model, writer):
    """
    Log the dynamic model parameters as a markdown-style table to TensorBoard.
    """
    # Log dynamic parameters
    arg_summary = "| **Parameter** | **Value** |\n|---|---|\n"
    for key, value in model.dynamics.items():
        if isinstance(value, np.ndarray):
            value_str = one_line_array(value)
        else:
            value_str = str(value)
        arg_summary += f"| {key} | {value_str} |\n"
    
    writer.add_text('Dynamic parameters', arg_summary)

    # Log survival parameters
    arg_summary = "| **Parameter** | **Value** |\n|---|---|\n"
    for key, value in model.survival.items():
        if isinstance(value, np.ndarray):
            value_str = one_line_array(value)
        else:
            value_str = str(value)
        arg_summary += f"| {key} | {value_str} |\n"
    
    writer.add_text('Survival/death parameters', arg_summary)

    # Log birth model parameters
    if not args.fixed_birth:
        arg_summary = "| **Birth model Parameter** | **Value** |\n|---|---|\n"
        for i, birth in enumerate(model.birth):
            arg_summary += f"| **Birth model {i+1}** | |\n"
            for key, value in birth.items():
                if isinstance(value, np.ndarray):
                    value_str = one_line_array(value)
                else:
                    value_str = str(value)
                arg_summary += f"| {key} | {value_str} |\n"
        
        writer.add_text('Birth model parameters', arg_summary)

    # Log sensor parameters
    arg_summary = "| **Sensor parameter** | **Value** |\n|---|---|\n"
    for i, sensor in enumerate(model.sensors):
        arg_summary += f"| **Sensor {i+1}** | |\n"
        for key, value in sensor.items():
            if isinstance(value, np.ndarray):
                value_str = one_line_array(value)
            else:
                value_str = str(value)
            arg_summary += f"| {key} | {value_str} |\n"
    
    writer.add_text('Sensor parameters', arg_summary)

def plot_sensor_positions(model, writer):
    """
    Plot the sensor positions and ranges and log the plot to TensorBoard.
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # Increase figure size and DPI for higher resolution
    ax.set_title('Sensor Positions and Ranges')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Different colors for different sensors
    colors = ['r', 'g', 'b', 'm']

    for i, sensor in enumerate(model.sensors):
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
    writer.add_figure('Sensor Positions and Ranges', fig)

    # Close the plot to free memory
    plt.close(fig)
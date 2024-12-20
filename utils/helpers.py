import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    
    writer.add_text('Survival and Death parameters', arg_summary)

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
    fig, ax = plt.subplots(figsize=(12, 10), dpi=250)
    ax.set_title('Sensor Positions and Ranges')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Create color palette based on number of sensors
    n_sensors = len(model.sensors)
    colors = create_color_palette(n_sensors)

    for i, sensor in enumerate(model.sensors):
        pos = sensor['X']
        range_c = sensor['range_c']
        color = colors[i]

        # Plot sensor position with different marker styles based on sensor type
        if sensor['type'] == 'brg':
            marker = 'o'
            label = f'Bearing Sensor {i+1}'
        else:  # brg_rng
            marker = 's'
            label = f'Range-Bearing Sensor {i+1}'

        # Plot sensor position
        ax.plot(pos[0], pos[1], marker, color=color, markersize=8, label=label)
        ax.text(pos[0], pos[1], f'S{i+1}', fontsize=9, ha='right', va='bottom')

        # Plot sensor range with increased spacing
        if len(range_c.shape) > 1:  # For range-bearing sensors
            theta = np.linspace(range_c[0, 0], range_c[0, 1], 50)
            r = np.linspace(range_c[1, 0], range_c[1, 1], 50)
            theta_grid, r_grid = np.meshgrid(theta, r)
            x_grid = r_grid * np.cos(theta_grid) + pos[0]
            y_grid = r_grid * np.sin(theta_grid) + pos[1]
            ax.plot(x_grid, y_grid, color=color, alpha=0.3)
        else:  # For bearing-only sensors
            theta = np.linspace(range_c[0], range_c[1], 100)
            r = np.ones_like(theta) * 4000  # Fixed range for visualization
            x = r * np.cos(theta) + pos[0]
            y = r * np.sin(theta) + pos[1]
            ax.plot(x, y, color=color, alpha=0.3)

    # Place legend at the bottom
    ax.legend(bbox_to_anchor=(0., -.25, 1., .102), 
             loc='upper left', ncol=4, mode="expand", borderaxespad=0.)

    ax.grid(True)
    plt.axis('equal')

    # Adjust layout to make room for legend at bottom
    plt.subplots_adjust(bottom=0.2)

    # Log the plot to TensorBoard
    writer.add_figure('Sensor Positions and Ranges', fig)

    # Close the plot to free memory
    plt.close(fig)

def create_color_palette(n_colors):
    """
    Create a color palette with n distinct colors
    """
    if n_colors <= 10:
        # Use Tableau Colors for small number of sensors
        colors = list(mcolors.TABLEAU_COLORS.values())
    elif n_colors <= 20:
        # Use combination of Tableau and CSS4 colors
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[:10]
    else:
        # Generate colors using HSV color space
        HSV_tuples = [(x/n_colors, 0.8, 0.9) for x in range(n_colors)]
        colors = list(map(lambda x: mcolors.hsv_to_rgb(x), HSV_tuples))
    
    return colors
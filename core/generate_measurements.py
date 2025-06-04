import os
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import create_color_palette
from utils.dataset import save_measurements

def gen_measurements(args, sensors, truth, rng=None, seed=None):
    """
    Generate measurements for all sensors.
    
    Args:
        args: Arguments from parse_args
        sensors: List of sensor models
        truth: Ground truth data. Contains:
            - K: Number of time steps 
            - X: List of target states for each time step
            - N: Number of targets at each time step
            - L: Number of births at each time step
            - I: Track IDs at each time step
            - W: Target weights at each time step
        rng: Random number generator (Matlab_RNG instance), optional
            If None, will use numpy's default random generator
        seed: Random seed for reproducible results, optional
            If provided, unique seeds will be generated for each time step and sensor
    
    Returns:
        meas: Dictionary containing measurements and parameters
            - K: Number of time steps
            - Z: List of measurements for each sensor at each time step
            - P_D: Detection probability for each sensor at each time step
            - lambda_c: Expected number of clutter returns for each sensor
    """
    num_sensors = len(sensors)
    meas = {
        'K': args.K,
        'Z': [[[] for _ in range(num_sensors)] for _ in range(args.K)],
        'P_D': np.zeros((args.K, num_sensors)),
        'lambda_c': np.zeros((args.K, num_sensors))
    }
    
    for k in range(args.K):
        for s in range(num_sensors):
            # Generate unique seed for this k and s if seed is provided
            if seed is not None:
                # Different seeds for different operations within the same k,s
                seed_pd = seed + k*1000 + s*100  # For detection probability
                seed_meas = seed + k*1000 + s*100 + 1  # For measurements
                seed_nc = seed + k*1000 + s*100 + 2  # For number of clutter
                seed_clutter = seed + k*1000 + s*100 + 3  # For clutter generation
            else:
                seed_pd = seed_meas = seed_nc = seed_clutter = None

            if truth['N'][k] > 0:
                # Get P_D for first-half or second-half
                sensor = sensors[s]
                P_D = sensor.P_D_rng[0]
                
                # Generate detection indicators using provided RNG or numpy
                if rng is not None:
                    idx = rng.rand(1, truth['N'][k], seed=seed_pd) <= P_D  # Matlab_RNG
                    idx = idx.flatten()
                else:
                    if seed_pd is not None:
                        np.random.seed(seed_pd)
                    idx = np.random.rand(truth['N'][k]) <= P_D  # NumPy random
                meas['P_D'][k,s] = P_D
                
                # Generate measurements for detected targets
                if np.any(idx):
                    selected_cols = np.where(idx)[0] 
                    meas['Z'][k][s] = gen_MS_observation(sensors, s, truth['X'][k][:, selected_cols], 'noise', rng, seed_meas)
                else:
                    meas['Z'][k][s] = np.array([])
            
            # Get clutter rate for first-half or second-half
            sensor = sensors[s]
            lambda_c = sensor.lambda_c_rng[0] if k < 50 else sensor.lambda_c_rng[1]
            meas['lambda_c'][k,s] = lambda_c
            
            # Generate clutter using provided RNG or numpy
            if rng is not None:
                N_c = rng.poisson(lambda_c, seed=seed_nc)  # Matlab_RNG
            else:
                if seed_nc is not None:
                    np.random.seed(seed_nc)
                N_c = np.random.poisson(lambda_c)  # NumPy random
            
            if N_c > 0:
                range_c = np.array(sensor.range_c)
                
                # Handle both 1D and 2D range_c
                if range_c.ndim == 1:
                    # For 1D measurements (e.g., bearing only)
                    if rng is not None:
                        C = range_c[0] + (range_c[1] - range_c[0]) * rng.rand(1, N_c, seed=seed_clutter)  # Matlab_RNG
                    else:
                        if seed_clutter is not None:
                            np.random.seed(seed_clutter)
                        C = np.random.uniform(range_c[0], range_c[1], size=(1, N_c))  # NumPy random
                else:
                    # For 2D measurements (e.g., bearing-range)
                    if rng is not None:
                        C = np.tile(range_c[:,0][:,None], [1, N_c]) + \
                            np.diag(range_c @ [-1, 1]) @ \
                            rng.rand(sensor.z_dim, N_c, seed=seed_clutter)  # Matlab_RNG
                    else:
                        if seed_clutter is not None:
                            np.random.seed(seed_clutter)
                        C = np.tile(range_c[:,0][:,None], [1, N_c]) + \
                            np.diag(range_c @ [-1, 1]) @ \
                            np.random.rand(sensor.z_dim, N_c)  # NumPy random
                # Combine target measurements and clutter
                if len(meas['Z'][k][s]) > 0:
                    # Ensure both matrices have the same number of rows for hstack
                    if isinstance(meas['Z'][k][s], np.ndarray):
                        # If measurement is a vector, reshape it to a matrix
                        if meas['Z'][k][s].ndim == 1:
                            meas['Z'][k][s] = meas['Z'][k][s].reshape(-1, 1)
                        # If clutter is a vector, reshape it to match measurement dimension
                        if C.ndim == 1:
                            C = C.reshape(-1, 1)
                    meas['Z'][k][s] = np.hstack([meas['Z'][k][s], C])
                else:
                    # If no measurements, ensure clutter has correct shape
                    if C.ndim == 1:
                        C = C.reshape(-1, 1)
                    meas['Z'][k][s] = C
    
    # Save measurements if dataset saving is enabled
    if hasattr(args, 'save_dataset') and args.save_dataset:
        scenario_name = getattr(args, 'scenario', 'default')
        save_measurements(meas, args.dataset_dir, scenario_name)

    return meas

def gen_MS_observation(sensors, s, X, W, rng=None, seed=None):
    """
    Generate observations for multiple sensors using sensor class methods.
    
    Args:
        sensors: List of sensor objects
        s: Sensor index
        X: Target states matrix (state_dim x num_targets)
        W: Noise type ('noise', 'noiseless') or noise matrix
        rng: Random number generator (Matlab_RNG instance), optional
        seed: Random seed for reproducible results, optional
    
    Returns:
        Z: Measurement matrix (z_dim x num_targets)
    """
    if X.size == 0:
        return np.array([])
    
    sensor = sensors[s]
    
    if W == 'noise':
        return sensor.generate_measurement(X, add_noise=True, rng=rng, seed=seed)
    elif W == 'noiseless':
        return sensor.generate_measurement(X, add_noise=False, rng=rng, seed=seed)
    else:
        # W is a noise matrix
        Z = sensor.generate_measurement(X, add_noise=False, rng=rng, seed=seed)
        return Z + W

def plot_measurements(args, truth, measurements, sensors, start_k, end_k, writer):
    """
    Plot measurements and ground truth from start_k to end_k.
    
    Args:
        args: Arguments from parse_args containing configuration parameters
        truth: Ground truth data dictionary
        measurements: Measurements dictionary
        sensors: List of sensor objects
        start_k: Start time step for plotting
        end_k: End time step for plotting
        writer: TensorboardX SummaryWriter for logging video
    
    Returns:
        None: Function saves video to TensorBoard and cleans up memory
    """
    from io import BytesIO
    from PIL import Image
    import gc

    frames = []
    sensor_colors = create_color_palette(len(sensors))
    
    for k in range(start_k, end_k):
        fig = plt.figure(figsize=(10, 10), dpi=250)
        ax = plt.gca()
        
        # Plot ground truth
        if truth['N'][k] > 0:
            ax.scatter(truth['X'][k][0,:], truth['X'][k][2,:],
                      c='k', marker='x', s=100,
                      label='Ground Truth')
        
        # Plot sensor positions and their measurements
        for s in range(len(sensors)):
            sensor = sensors[s]
            sensor_pos = sensor.position
            sensor_type = sensor.sensor_type
            
            # Plot sensor position
            ax.scatter(sensor_pos[0], sensor_pos[1],
                      c=[sensor_colors[s]], marker='*', s=200,
                      label=f'Sensor {s+1}')
            
            # Plot measurements for this sensor
            if len(measurements['Z'][k][s]) > 0:
                if sensor_type in ['pos', 'pos_3D']:
                    # Direct position measurements
                    ax.scatter(measurements['Z'][k][s][0,:], 
                             measurements['Z'][k][s][1,:],
                             c=[sensor_colors[s]], marker='o', s=64,
                             alpha=0.5)
                elif sensor_type in ['brg_rng', 'brg_rng_rngrt']:
                    # Convert polar to Cartesian for plotting
                    for i in range(measurements['Z'][k][s].shape[1]):
                        brg = measurements['Z'][k][s][0,i]
                        range_val = measurements['Z'][k][s][1,i]
                        x = sensor_pos[0] + range_val * np.cos(brg)
                        y = sensor_pos[1] + range_val * np.sin(brg)
                        ax.scatter(x, y,
                                 c=[sensor_colors[s]], marker='o', s=64,
                                 alpha=0.5)
                elif sensor_type == 'brg':
                    # Plot bearing measurements as lines from sensor position
                    for i in range(measurements['Z'][k][s].shape[1]):
                        brg = measurements['Z'][k][s][0,i]
                        line_length = 4000
                        x = sensor_pos[0] + line_length * np.cos(brg)
                        y = sensor_pos[1] + line_length * np.sin(brg)
                        ax.plot([sensor_pos[0], x],
                               [sensor_pos[1], y],
                               c=sensor_colors[s], alpha=0.3)
        
        # Set plot properties
        ax.set_xlim(-4000, 4000)
        ax.set_ylim(-4000, 4000)
        ax.grid(True)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Time Step {k}')
        ax.legend(loc='upper right')
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        frame = np.array(image)
        frames.append(frame)
        buf.close()
        plt.close(fig)

    # Convert frames to video format: [T, C, H, W]
    video = np.stack(frames)  # [T, H, W, C]
    video = video.transpose((0, 3, 1, 2))  # [T, C, H, W]

    # Log video to TensorBoardX
    writer.add_video('Measurements', video[np.newaxis], fps=5.0)

    # Clean up frames and video
    del frames
    del video
    gc.collect()  # Trigger garbage collection to free memory
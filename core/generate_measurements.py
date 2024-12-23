import os
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import create_color_palette

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
                # Get P_D (first-half, second-half)
                P_D = sensors[s]['P_D_rng'][0]
                
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
            
            # Get clutter rate (first-half, second-half)
            lambda_c = sensors[s]['lambda_c_rng'][0] if k < 50 else sensors[s]['lambda_c_rng'][1]
            meas['lambda_c'][k,s] = lambda_c
            
            # Generate clutter using provided RNG or numpy
            if rng is not None:
                N_c = rng.poisson(lambda_c, seed=seed_nc)  # Matlab_RNG
            else:
                if seed_nc is not None:
                    np.random.seed(seed_nc)
                N_c = np.random.poisson(lambda_c)  # NumPy random
            
            if N_c > 0:
                range_c = sensors[s]['range_c']
                
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
                            rng.rand(sensors[s]['z_dim'], N_c, seed=seed_clutter)  # Matlab_RNG
                    else:
                        if seed_clutter is not None:
                            np.random.seed(seed_clutter)
                        C = np.tile(range_c[:,0][:,None], [1, N_c]) + \
                            np.diag(range_c @ [-1, 1]) @ \
                            np.random.rand(sensors[s]['z_dim'], N_c)  # NumPy random
                # Combine target measurements and clutter
                if len(meas['Z'][k][s]) > 0:
                    # Ensure both matrices have the same number of rows for hstack
                    if isinstance(meas['Z'][k][s], np.ndarray):
                        # If measurement is a vector, reshape it to a 2x1 matrix
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
    
    return meas

def gen_MS_observation(sensors, s, X, W, rng=None, seed=None):
    """
    Generate observations for multiple sensors.
    
    Args:
        sensors: List of sensor models, each model contains:
            - type: Sensor type ('brg', 'rng', 'brg_rr', etc.)
            - z_dim: Measurement dimension
            - R: Measurement noise covariance
            - X: Sensor position [x, y] or [x, y, z]
        s: Sensor index
        X: Target states matrix (state_dim × num_targets)
        W: Noise type ('noise', 'noiseless') or noise matrix
        rng: Random number generator (Matlab_RNG instance), optional
            If None, will use numpy's default random generator
        seed: Random seed for reproducible results, optional
            If provided, will be used for noise generation
    
    Returns:
        Z: Measurement matrix (z_dim × num_targets)
    """
    if not isinstance(W, np.ndarray):
        if W == 'noise':
            if rng is not None:
                W = rng.multivariate_normal(
                    mean=np.zeros(sensors[s]['z_dim']), 
                    cov=sensors[s]['R'], 
                    size=X.shape[1],
                    seed=seed).T  # Matlab_RNG
            else:
                if seed is not None:
                    np.random.seed(seed)
                if np.isscalar(sensors[s]['R']) or sensors[s]['R'].size == 1:
                    # For scalar covariance (e.g., bearing-only sensors)
                    W = np.random.normal(0, np.sqrt(float(sensors[s]['R'])), 
                                    size=(1, X.shape[1]))
                else:
                    # For matrix covariance
                    W = np.random.multivariate_normal(
                        mean=np.zeros(sensors[s]['z_dim']), 
                        cov=sensors[s]['R'], 
                        size=X.shape[1]).T  # NumPy random
        elif W == 'noiseless':
            W = np.zeros((sensors[s]['R'].shape[0], X.shape[1]))

    if X.size == 0:
        return np.array([])
    
    sensor_type = sensors[s]['type']
    
    if sensor_type == 'brg':        # Bearing-only sensor
        #! TODO: need to check arctan(y, x) vs arctan(x, y)
        Z = np.arctan2(X[0,:] - sensors[s]['X'][0],     # x (first parameter in MATLAB)
                       X[2,:] - sensors[s]['X'][1])     # y (second parameter in MATLAB)
        
        # Handle vector case for bearing-only measurements
        if isinstance(W, np.ndarray) and W.ndim == 1:
            Z = Z.flatten()
        else:
            Z = Z.reshape(1, -1)  # Ensure 2D for matrix case
        Z = Z + W
        Z = np.mod(Z, 2*np.pi)

    elif sensor_type == 'rng':      # Range-only sensor
        Z = np.sqrt((sensors[s]['X'][0] - X[0,:])**2 + 
                    (sensors[s]['X'][1] - X[2,:])**2)
        # Handle vector case for range-only measurements
        if isinstance(W, np.ndarray) and W.ndim == 1:
            Z = Z.flatten()
        else:
            Z = Z.reshape(1, -1)  # Ensure 2D for matrix case
        Z = Z + W
        
    elif sensor_type == 'brg_rr':       # Bearing and Range Rate sensors: Doppler radar systems, Air traffic control radar
        relpos = X[[0,2],:] - sensors[s]['X'][:,None]
        relvel = X[[1,3],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        
        if isinstance(W, np.ndarray) and W.ndim == 1:
            # Vector case
            Z = np.zeros(2)
            Z[0] = np.arctan2(relpos[0,0], relpos[1,0])
            Z[1] = np.sum(relpos[:,0] * relvel[:,0]) / rng[0]
        else:
            # Matrix case
            Z = np.zeros((2, X.shape[1]))
            Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])
            Z[1,:] = np.sum(relpos * relvel, axis=0) / rng
        
        Z = Z + W
        if Z.ndim > 1:
            Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        else:
            Z[0] = np.mod(Z[0], 2*np.pi)
        
    elif sensor_type == 'pos':      # x and y coordinates
        if isinstance(W, np.ndarray) and W.ndim == 1:
            # Vector case
            Z = np.zeros(2)
            Z[0:2] = X[[0,2],0]
        else:
            # Matrix case
            Z = np.zeros((2, X.shape[1]))
            Z[0:2,:] = X[[0,2],:]
        Z = Z + W
        
    elif sensor_type == 'pos_3D':       # x, y, and z coordinates
        if isinstance(W, np.ndarray) and W.ndim == 1:
            # Vector case
            Z = np.zeros(3)
            Z[0:3] = X[[0,2,4],0]
        else:
            # Matrix case
            Z = np.zeros((3, X.shape[1]))
            Z[0:3,:] = X[[0,2,4],:]
        Z = Z + W
        
    elif sensor_type == 'brg_rng':      # Bearing-range sensor
        relpos = X[[0,2],:] - sensors[s]['X'][:,None]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        
        if isinstance(W, np.ndarray) and W.ndim == 1:
            # Vector case
            Z = np.zeros(2)
            Z[0] = np.arctan2(relpos[1,0], relpos[0,0])
            Z[1] = rng[0]
        else:
            # Matrix case
            Z = np.zeros((2, X.shape[1]))
            Z[0,:] = np.arctan2(relpos[1,:], relpos[0,:])
            Z[1,:] = rng
            
        Z = Z + W
        if Z.ndim > 1:
            Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        else:
            Z[0] = np.mod(Z[0], 2*np.pi)
            
    elif sensor_type == 'az_el_rng':        # Azimuth, Elevation, and Range sensor: 3D scanning LiDAR
        relpos = X[[0,2,5],:] - sensors[s]['X'][:,None]
        relvel = X[[1,3,6],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        
        if isinstance(W, np.ndarray) and W.ndim == 1:
            # Vector case
            Z = np.zeros(4)
            xy_rng = relpos[0:2,0]
            xy_rng = np.sqrt(np.sum(xy_rng**2))
            Z[0] = np.arctan2(relpos[0,0], relpos[1,0])
            Z[1] = np.arctan2(xy_rng, relpos[2,0])
            Z[2] = rng[0]
            Z[3] = np.sum(relpos[:,0] * relvel[:,0]) / rng[0]
        else:
            # Matrix case
            Z = np.zeros((4, X.shape[1]))
            xy_rng = relpos[0:2,:]
            xy_rng = np.sqrt(np.sum(xy_rng**2, axis=0))
            Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])
            Z[1,:] = np.arctan2(xy_rng, relpos[2,:])
            Z[2,:] = rng
            Z[3,:] = np.sum(relpos * relvel, axis=0) / rng
            
        Z = Z + W
        if Z.ndim > 1:
            Z[0,:] = np.mod(Z[0,:] + np.pi, 2*np.pi) - np.pi
            Z[1,:] = np.mod(Z[1,:] + np.pi, 2*np.pi) - np.pi
        else:
            Z[0] = np.mod(Z[0] + np.pi, 2*np.pi) - np.pi
            Z[1] = np.mod(Z[1] + np.pi, 2*np.pi) - np.pi
            
    elif sensor_type == 'brg_rng_rngrt':        # Bearing, Range, and Range Rate sensor: Modern military tracking radar
        relpos = X[[0,2],:] - sensors[s]['X'][:,None]
        relvel = X[[1,3],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        
        if isinstance(W, np.ndarray) and W.ndim == 1:
            # Vector case
            Z = np.zeros(3)
            Z[0] = np.arctan2(relpos[0,0], relpos[1,0])
            Z[1] = rng[0]
            Z[2] = np.sum(relpos[:,0] * relvel[:,0]) / rng[0]
        else:
            # Matrix case
            Z = np.zeros((3, X.shape[1]))
            Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])
            Z[1,:] = rng
            Z[2,:] = np.sum(relpos * relvel, axis=0) / rng
            
        Z = Z + W
        if Z.ndim > 1:
            Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        else:
            Z[0] = np.mod(Z[0], 2*np.pi)
            
    return Z

def plot_measurements(args, truth, measurements, sensors, start_k, end_k, writer):
    """
    Plot measurements and ground truth from start_k to end_k.
    
    Args:
        args: Arguments from parse_args
        truth: Ground truth data dictionary
        measurements: Measurements dictionary
        sensors: List of sensor models
        start_k: Start time step
        end_k: End time step
        writer: tensorboardX SummaryWriter
    """
    from io import BytesIO
    from PIL import Image
    import gc

    frames = []
    # Create color palette based on number of sensors
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
            # Plot sensor position
            ax.scatter(sensors[s]['X'][0], sensors[s]['X'][1],
                      c=[sensor_colors[s]], marker='*', s=200,
                      label=f'Sensor {s+1}')
            
            # Plot measurements for this sensor
            if len(measurements['Z'][k][s]) > 0:
                if sensors[s]['type'] in ['pos', 'pos_3D']:
                    # Direct position measurements
                    ax.scatter(measurements['Z'][k][s][0,:], 
                             measurements['Z'][k][s][1,:],
                             c=[sensor_colors[s]], marker='o', s=64,
                             alpha=0.5)
                elif sensors[s]['type'] in ['brg_rng', 'brg_rng_rngrt']:
                    # Convert polar to Cartesian for plotting
                    for i in range(measurements['Z'][k][s].shape[1]):
                        brg = measurements['Z'][k][s][0,i]
                        rng = measurements['Z'][k][s][1,i]
                        x = sensors[s]['X'][0] + rng * np.cos(brg)
                        y = sensors[s]['X'][1] + rng * np.sin(brg)
                        ax.scatter(x, y,
                                 c=[sensor_colors[s]], marker='o', s=64,
                                 alpha=0.5)
                elif sensors[s]['type'] == 'brg':
                    # Plot bearing measurements as lines from sensor position
                    for i in range(measurements['Z'][k][s].shape[1]):
                        brg = measurements['Z'][k][s][0,i]
                        # Plot a line in the bearing direction
                        line_length = 4000  # Same as plot limits
                        x = sensors[s]['X'][0] + line_length * np.cos(brg)
                        y = sensors[s]['X'][1] + line_length * np.sin(brg)
                        ax.plot([sensors[s]['X'][0], x],
                               [sensors[s]['X'][1], y],
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
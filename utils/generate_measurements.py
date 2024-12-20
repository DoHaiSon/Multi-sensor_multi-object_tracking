import os
import numpy as np
import matplotlib.pyplot as plt

def gen_measurements(args, sensors, truth):
    """
    Generate measurements for all sensors.
    
    Args:
        sensors: List of sensor models
        truth: Ground truth data
        seed: Random seed (optional)
    
    Returns:
        meas: Dictionary containing measurements and parameters
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
            if truth['N'][k] > 0:
                # Get P_D (first-half, second-half)
                P_D = sensors[s]['P_D_rng'][0]
                
                # Generate detection indicators
                idx = np.random.rand(truth['N'][k]) <= P_D
                meas['P_D'][k,s] = P_D
                
                # Generate measurements for detected targets
                if np.any(idx):
                    meas['Z'][k][s] = gen_MS_observation(
                        sensors, s, truth['X'][k][:, idx], 'noise')
                else:
                    meas['Z'][k][s] = np.array([])
            
            # Get clutter rate (first-half, second-half)
            lambda_c = sensors[s]['lambda_c_rng'][0] if k < 50 else sensors[s]['lambda_c_rng'][1]
            meas['lambda_c'][k,s] = lambda_c
            
            # Generate clutter
            N_c = np.random.poisson(lambda_c)
            if N_c > 0:
                C = np.tile(sensors[s]['range_c'][:,0][:,None], [1, N_c]) + \
                    np.diag(sensors[s]['range_c'] @ [-1, 1]) @ \
                    np.random.rand(sensors[s]['z_dim'], N_c)
                
                # Combine target measurements and clutter
                if len(meas['Z'][k][s]) > 0:
                    meas['Z'][k][s] = np.hstack([meas['Z'][k][s], C])
                else:
                    meas['Z'][k][s] = C
    
    return meas

def gen_MS_observation(sensors, s, X, W):
    """
    Generate observations for multiple sensors.
    
    Args:
        sensors: List of sensor models, each model is a dictionary containing sensor parameters
        s: Sensor index
        X: Target states
        W: Noise type ('noise', 'noiseless') or noise matrix
    
    Returns:
        Z: Measurement matrix
    """
    if not isinstance(W, np.ndarray):
        if W == 'noise':
            # Generate multivariate normal noise
            W = np.random.multivariate_normal(
                mean=np.zeros(sensors[s]['z_dim']), 
                cov=sensors[s]['R'], 
                size=X.shape[1]).T
        elif W == 'noiseless':
            W = np.zeros((sensors[s]['R'].shape[0], X.shape[1]))
    
    if X.size == 0:
        return np.array([])
    
    sensor_type = sensors[s]['type']
    
    if sensor_type == 'brg':
        Z = np.mod(np.arctan2(X[0,:] - sensors[s]['X'][0], 
                             X[2,:] - sensors[s]['X'][1]) + W, 2*np.pi)
        
    elif sensor_type == 'rng':
        Z = np.sqrt((sensors[s]['X'][0] - X[0,:])**2 + 
                    (sensors[s]['X'][1] - X[2,:])**2) + W
        
    elif sensor_type == 'brg_rr':
        relpos = X[[0,2],:] - sensors[s]['X'][:,None]
        relvel = X[[1,3],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        Z = np.zeros((2, X.shape[1]))
        Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])
        Z[1,:] = np.sum(relpos * relvel, axis=0) / rng
        Z = Z + W
        Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        
    elif sensor_type == 'pos':
        Z = np.zeros((2, X.shape[1]))
        Z[0:2,:] = X[[0,2],:]
        Z = Z + W
        
    elif sensor_type == 'pos_3D':
        Z = np.zeros((3, X.shape[1]))
        Z[0:3,:] = X[[0,2,4],:]
        Z = Z + W
        
    elif sensor_type == 'brg_rng':
        relpos = X[[0,2],:] - sensors[s]['X'][:,None]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        Z = np.zeros((2, X.shape[1]))
        Z[0,:] = np.arctan2(relpos[1,:], relpos[0,:])
        Z[1,:] = rng
        Z = Z + W
        Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        
    elif sensor_type == 'az_el_rng':
        relpos = X[[0,2,5],:] - sensors[s]['X'][:,None]
        relvel = X[[1,3,6],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        Z = np.zeros((4, X.shape[1]))
        Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])
        xy_rng = relpos[0:2,:]
        xy_rng = np.sqrt(np.sum(xy_rng**2, axis=0))
        Z[1,:] = np.arctan2(xy_rng, relpos[2,:])
        Z[2,:] = rng
        Z[3,:] = np.sum(relpos * relvel, axis=0) / rng
        Z = Z + W
        Z[0,:] = np.mod(Z[0,:] + np.pi, 2*np.pi) - np.pi  # azimuth
        Z[1,:] = np.mod(Z[1,:] + np.pi, 2*np.pi) - np.pi  # elevation
        
    elif sensor_type == 'brg_rng_rngrt':
        relpos = X[[0,2],:] - sensors[s]['X'][:,None]
        relvel = X[[1,3],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        Z = np.zeros((3, X.shape[1]))
        Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])
        Z[1,:] = rng
        Z[2,:] = np.sum(relpos * relvel, axis=0) / rng
        Z = Z + W
        Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        
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
    import cv2

    frames = []
    sensor_colors = ['r', 'g', 'b', 'm']  # Colors for each sensor
    
    for k in range(start_k, end_k):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot ground truth
        if truth['N'][k] > 0:
            ax.scatter(truth['X'][k][0,:], truth['X'][k][2,:],
                      c='k', marker='x', s=100,
                      label='Ground Truth')
        
        # Plot sensor positions and their measurements
        for s in range(len(sensors)):
            # Plot sensor position
            ax.scatter(sensors[s]['X'][0], sensors[s]['X'][1],
                      c=sensor_colors[s], marker='*', s=200,
                      label=f'Sensor {s+1}')
            
            # Plot measurements for this sensor
            if len(measurements['Z'][k][s]) > 0:
                if sensors[s]['type'] in ['pos', 'pos_3D']:
                    # Direct position measurements
                    ax.scatter(measurements['Z'][k][s][0,:], 
                             measurements['Z'][k][s][1,:],
                             c=sensor_colors[s], marker='o', s=64,
                             alpha=0.5)
                elif sensors[s]['type'] in ['brg_rng', 'brg_rng_rngrt']:
                    # Convert polar to Cartesian for plotting
                    for i in range(measurements['Z'][k][s].shape[1]):
                        brg = measurements['Z'][k][s][0,i]
                        rng = measurements['Z'][k][s][1,i]
                        x = sensors[s]['X'][0] + rng * np.cos(brg)
                        y = sensors[s]['X'][1] + rng * np.sin(brg)
                        ax.scatter(x, y,
                                 c=sensor_colors[s], marker='o', s=64,
                                 alpha=0.5)
        
        # Set plot properties
        ax.set_xlim(-4000, 4000)
        ax.set_ylim(-4000, 4000)
        ax.grid(True)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Time Step {k}')
        ax.legend()
        
        # Convert plot to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)

        plt.close(fig)

    height, width = frames[0].shape[:2]
    video_path = os.path.join(args.log_dir, f'measurements_{start_k}_{end_k}.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
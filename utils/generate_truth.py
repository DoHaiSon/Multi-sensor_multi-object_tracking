import numpy as np
from utils.generate_new_state import gen_new_state
import matplotlib.pyplot as plt

def gen_truth(args, model):
    truth = {
        'K': args.K,  # length of data/number of scans
        'X': [None] * args.K,  # ground truth for states of targets
        'N': np.zeros(args.K, dtype=int),  # ground truth for number of targets
        'L': [None] * args.K,  # ground truth for labels of targets (k,i)
        'track_list': [None] * args.K,  # absolute index target identities (plotting)
        'total_tracks': 0  # total number of appearing tracks
    }

    nbirths = 18
    wturn = 2 * np.pi / 180

    xstart = np.array([
        [1000 + 3.8676, -10, 1500 - 11.7457, -10, wturn / 8],
        [-250 - 5.8857, 20, 1000 + 11.4102, 3, -wturn / 3],
        [-1500 - 7.3806, 11, 250 + 6.7993, 10, -wturn / 2],
        [-1500, 43, 1500, 0, 0],
        [250 - 3.8676, 11, 750 - 11.0747, 5, wturn / 4],
        [1000 + 3.8676, -8, 1500 - 11.7457, -10, wturn / 6],
        [-250, 15, 1000 + 11.4102, 3, -wturn / 3],
        [-250 + 7.3806, -12, 1000 - 6.7993, -12, wturn / 2],
        [1000, 0, 1500, -10, wturn / 4],
        [250, -50, 750, 0, -wturn / 4],
        [1000, -20, 1500, -30, -wturn / 6],
        [1300, -40, 700, 20, wturn / 8],
        [1000, -50, 1500, 0, -wturn / 4],
        [250, -40, 750, 25, wturn / 4],
        [1100, -40, 1300, -10, -wturn / 3],
        [500, -60, 750, 25, wturn / 4],
        [1200, -50, 1500, 0, -wturn / 4],
        [250, -40, 750, 30, wturn / 4]
    ]).T

    tbirth = np.array([1, 1, 10, 10, 20, 30, 30, 40, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80])
    tdeath = np.array([101, 101, 101, 66, 80, 70, 80, 101, 101, 80, 90, 101, 101, 101, 90, 101, 90, 101])

    # Generate the tracks
    for targetnum in range(nbirths):
        targetstate = xstart[:, targetnum]
        for k in range(tbirth[targetnum] - 1, min(tdeath[targetnum], truth['K'])):

            targetstate = gen_new_state(args, model, targetstate, 'noiseless')
            
            if truth['X'][k] is None:
                truth['X'][k] = targetstate.reshape(-1, 1)
            else:
                truth['X'][k] = np.hstack((truth['X'][k], targetstate.reshape(-1, 1)))
            if truth['track_list'][k] is None:
                truth['track_list'][k] = [targetnum + 1]
            else:
                truth['track_list'][k].append(targetnum + 1)
            truth['N'][k] += 1

    truth['total_tracks'] = nbirths

    return truth

def plot_truth(truth, t1, t2, writer, global_step=0):
    """
    Plot ground truth tracks and save to TensorBoard.
    
    Args:
        truth: Dictionary containing ground truth data
        t1: Start time
        t2: End time
        writer: TensorBoard writer
        global_step: Current step for TensorBoard logging
    """
    # Create figure with high DPI
    plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.gca()
    
    # Set axes properties
    ax.set_xlabel('X Position [m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position [m]', fontsize=14, fontweight='bold')
    ax.grid(True)
    ax.set_box_aspect(1)  # Equal aspect ratio
    
    # Extract tracks
    X_track, _, _ = extract_tracks(truth['X'], truth['track_list'], truth['total_tracks'])
    
    # If time window specified, slice the tracks
    if t2 is not None:
        X_track = X_track[:, t1-1:t2, :]  
    
    # Plot each track
    for i in range(X_track.shape[2]):
        # Find start and end indices (where track is not NaN)
        track_x = X_track[0, :, i]
        track_y = X_track[2, :, i]
        valid_indices = ~np.isnan(track_x)
        
        if np.any(valid_indices):
            start_idx = np.where(valid_indices)[0][0]
            end_idx = np.where(valid_indices)[0][-1]
            
            # Plot track line
            plt.plot(track_x[start_idx:end_idx+1], 
                    track_y[start_idx:end_idx+1], 
                    'k-')
            
            # Plot start point (green circle)
            plt.plot(track_x[start_idx], 
                    track_y[start_idx], 
                    'o', 
                    color='black',
                    markerfacecolor='green',
                    markersize=8)
            
            # Plot end point (red square)
            plt.plot(track_x[end_idx], 
                    track_y[end_idx], 
                    's', 
                    color='black',
                    markerfacecolor='red',
                    markersize=8)
    
    # Add title
    plt.title('Ground Truth Tracks', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Log to TensorBoard
    writer.add_figure('Ground Truth Tracks', plt.gcf(), global_step)
    
    # Close the figure to free memory
    plt.close()

def extract_tracks(X, track_list, total_tracks):
    """
    Convert list of states and track lists to 3D array of tracks.
    """
    K = len(X)
    
    # Find first non-empty X to get state dimension
    x_dim = 0
    k = K - 1
    while x_dim == 0 and k >= 0:
        if X[k] is not None and len(X[k]) > 0:
            x_dim = X[k].shape[0]
        k -= 1

    # Initialize output arrays
    X_track = np.full((x_dim, K, total_tracks), np.nan)
    k_birth = np.zeros(total_tracks)
    k_death = np.zeros(total_tracks)
    
    max_idx = 0
    for k in range(K):
        if X[k] is not None and len(X[k]) > 0:
            current_tracks = track_list[k]
            if current_tracks:  # if not empty
                curr_tracks_array = np.array(current_tracks) - 1  # Convert to 0-based indexing
                X_track[:, k, curr_tracks_array] = X[k]
                
                # Check for new targets
                if max(current_tracks) > max_idx:
                    idx = np.where(np.array(current_tracks) > max_idx)[0]
                    k_birth[np.array(current_tracks)[idx] - 1] = k
                max_idx = max(current_tracks)
                k_death[np.array(current_tracks) - 1] = k
    
    return X_track, k_birth, k_death
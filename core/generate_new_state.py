import numpy as np

def gen_new_state(args, model, target_state, noise_type, rng=None):
    """
    Generate new state based on the model and given noise type.

    Parameters:
    model (dict): Dictionary containing model parameters
    target_state (np.ndarray): Current state of the target.
    noise_type (str): Type of noise to apply ('noise' or 'noiseless').
    rng (Matlab_RNG): Random number generator, optional
        If None, will use numpy's default random generator

    Returns:
    np.ndarray: New state with or without noise.
    """
    # Ensure target_state is a 2D array
    if target_state.ndim == 1:
        target_state = target_state.reshape(-1, 1)

    if noise_type == 'noise':
        # Use provided RNG if available
        if rng is not None:
            V = np.dot(model['B'], rng.randn(model['B'].shape[1], target_state.shape[1]))
        else:
            V = np.dot(model['B'], np.random.randn(model['B'].shape[1], target_state.shape[1]))
    elif noise_type == 'noiseless':
        V = np.zeros((model['B'].shape[0], target_state.shape[1]))
    else:
        raise ValueError("Invalid noise_type. Use 'noise' or 'noiseless'.")

    if target_state.size == 0:
        return np.array([])

    X = np.zeros_like(target_state)
   
    if args.CT:
        X = CT_model(model, target_state)
    else:
        X = CV_model(model, target_state)

    if model['B2'].shape[1] != V.shape[0]:
        raise ValueError(f"Shapes not aligned: model['B2'].shape = {model['B2'].shape}, V.shape = {V.shape}")

    X_noise = X + np.dot(model['B2'], V)
    return X_noise

def CT_model(model, Xd):
    """
    Constant Turn (CT) model.

    Parameters:
    model (dict): Dictionary containing model parameters including 'T'.
    Xd (np.ndarray): Current state of the target.

    Returns:
    np.ndarray: New state using the CT model.
    """
    X = np.zeros_like(Xd)
    L = Xd.shape[1]
    T = model['T']
    omega = Xd[4, :]
    tol = 1e-10

    sin_omega_T = np.sin(omega * T)
    cos_omega_T = np.cos(omega * T)
    a = T * np.ones(L)
    b = np.zeros(L)
    idx = np.abs(omega) > tol
    a[idx] = sin_omega_T[idx] / omega[idx]
    b[idx] = (1 - cos_omega_T[idx]) / omega[idx]

    X[0, :] = Xd[0, :] + a * Xd[1, :] - b * Xd[3, :]
    X[1, :] = cos_omega_T * Xd[1, :] - sin_omega_T * Xd[3, :]
    X[2, :] = b * Xd[1, :] + Xd[2, :] + a * Xd[3, :]
    X[3, :] = sin_omega_T * Xd[1, :] + cos_omega_T * Xd[3, :]
    X[4, :] = Xd[4, :]

    return X

def CV_model(model, Xd):
    """
    Constant Velocity (CV) model.

    Parameters:
    model (dict): Dictionary containing model parameters including 'T'.
    Xd (np.ndarray): Current state of the target.

    Returns:
    np.ndarray: New state using the CV model.
    """
    T = model['T']
    X = np.zeros_like(Xd)

    X[0, :] = Xd[0, :] + T * Xd[1, :]
    X[1, :] = Xd[1, :]
    X[2, :] = Xd[2, :] + T * Xd[3, :]
    X[3, :] = Xd[3, :]
    X[4, :] = Xd[4, :]

    return X
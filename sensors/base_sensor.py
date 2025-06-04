import numpy as np
from abc import ABC, abstractmethod

class BaseSensor(ABC):
    """Base class for all sensors."""
    
    def __init__(self, sensor_id, position, P_D_rng, lambda_c_rng, R, range_c=None):
        """
        Initialize base sensor.
        
        Args:
            sensor_id: Unique sensor identifier
            position: Sensor position [x, y] or [x, y, z]
            P_D_rng: Detection probability range [P_D_first_half, P_D_second_half]
            lambda_c_rng: Clutter rate range [lambda_c_first_half, lambda_c_second_half]
            R: Measurement noise covariance matrix
            range_c: Clutter range
        """
        self.sensor_id = sensor_id
        self.position = np.array(position)
        self.P_D_rng = P_D_rng
        self.lambda_c_rng = lambda_c_rng
        self.R = np.array(R) if not isinstance(R, np.ndarray) else R
        self.range_c = np.array(range_c) if range_c is not None else self._default_range_c()
        
    @property
    @abstractmethod
    def sensor_type(self):
        """Return sensor type string."""
        pass
    
    @property
    @abstractmethod
    def z_dim(self):
        """Return measurement dimension."""
        pass
    
    @abstractmethod
    def _default_range_c(self):
        """Return default clutter range for this sensor type."""
        pass
    
    @abstractmethod
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """
        Generate measurements from target states.
        
        Args:
            target_states: Target states matrix (state_dim x num_targets)
            add_noise: Whether to add measurement noise
            rng: Random number generator
            seed: Random seed
            
        Returns:
            Measurement matrix (z_dim x num_targets)
        """
        pass
    
    def generate_noise(self, num_measurements, rng=None, seed=None):
        """
        Generate measurement noise.
        
        Args:
            num_measurements: Number of measurements to generate noise for
            rng: Random number generator (Matlab_RNG instance), optional
            seed: Random seed for reproducible results, optional
        
        Returns:
            np.ndarray: Noise matrix with shape (z_dim, num_measurements)
        """
        if rng is not None:
            # Use MATLAB-compatible RNG
            if self.R.ndim == 0:  # Scalar variance
                noise_std = np.sqrt(self.R)
                return rng.normal(0, noise_std, (self.z_dim, num_measurements), seed=seed)
            else:  # Covariance matrix
                # Generate multivariate normal noise
                noise = np.zeros((self.z_dim, num_measurements))
                for i in range(num_measurements):
                    sample_seed = seed + i if seed is not None else None
                    noise[:, i] = rng.multivariate_normal(
                        np.zeros(self.z_dim), 
                        self.R, 
                        seed=sample_seed
                    )
                return noise
        else:
            # Use NumPy random
            if seed is not None:
                np.random.seed(seed)
            
            if self.R.ndim == 0:  # Scalar variance
                noise_std = np.sqrt(self.R)
                return np.random.normal(0, noise_std, (self.z_dim, num_measurements))
            else:  # Covariance matrix
                return np.random.multivariate_normal(
                    np.zeros(self.z_dim), 
                    self.R, 
                    num_measurements
                ).T
    
    def generate_clutter(self, num_clutter, rng=None, seed=None):
        """Generate clutter measurements."""
        if num_clutter == 0:
            return np.array([]).reshape(self.z_dim, 0)
            
        if self.range_c.ndim == 1:
            # For 1D measurements
            if rng is not None:
                C = self.range_c[0] + (self.range_c[1] - self.range_c[0]) * \
                    rng.rand(1, num_clutter, seed=seed)
            else:
                if seed is not None:
                    np.random.seed(seed)
                C = np.random.uniform(self.range_c[0], self.range_c[1], size=(1, num_clutter))
        else:
            # For multi-dimensional measurements
            if rng is not None:
                C = np.tile(self.range_c[:,0][:,None], [1, num_clutter]) + \
                    np.diag(self.range_c @ [-1, 1]) @ \
                    rng.rand(self.z_dim, num_clutter, seed=seed)
            else:
                if seed is not None:
                    np.random.seed(seed)
                C = np.tile(self.range_c[:,0][:,None], [1, num_clutter]) + \
                    np.diag(self.range_c @ [-1, 1]) @ \
                    np.random.rand(self.z_dim, num_clutter)
        
        return C
    
    def to_dict(self):
        """Convert sensor to dictionary format for compatibility."""
        return {
            'type': self.sensor_type,
            'z_dim': self.z_dim,
            'X': self.position,
            'P_D_rng': self.P_D_rng,
            'lambda_c_rng': self.lambda_c_rng,
            'R': self.R,
            'range_c': self.range_c
        }

import numpy as np
from .base_sensor import BaseSensor

class BearingRangeSensor(BaseSensor):
    """Bearing-range sensor implementation."""
    
    @property
    def sensor_type(self):
        return 'brg_rng'
    
    @property
    def z_dim(self):
        return 2
    
    def _default_range_c(self):
        return np.array([[0, 2*np.pi], [0, 5000]])
    
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """Generate bearing-range measurements."""
        if target_states.size == 0:
            return np.array([])
        
        # Calculate relative position
        relpos = target_states[[0,2],:] - self.position[:,None]
        range_val = np.sqrt(np.sum(relpos**2, axis=0))
        
        Z = np.zeros((2, target_states.shape[1]))
        Z[0,:] = np.arctan2(relpos[1,:], relpos[0,:])  # Bearing
        Z[1,:] = range_val  # Range
        
        # Add noise if requested
        if add_noise:
            noise = self.generate_noise(target_states.shape[1], rng, seed)
            Z = Z + noise
        
        # Wrap bearing to [0, 2π]
        Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        
        return Z

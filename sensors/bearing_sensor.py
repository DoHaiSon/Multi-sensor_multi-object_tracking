import numpy as np
from .base_sensor import BaseSensor

class BearingSensor(BaseSensor):
    """Bearing-only sensor implementation."""
    
    @property
    def sensor_type(self):
        return 'brg'
    
    @property
    def z_dim(self):
        return 1
    
    def _default_range_c(self):
        return np.array([0, 2*np.pi])
    
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """Generate bearing-only measurements."""
        if target_states.size == 0:
            return np.array([])
        
        # Calculate relative position
        relpos = target_states[[0,2],:] - self.position[:,None]
        
        Z = np.zeros((1, target_states.shape[1]))
        Z[0,:] = np.arctan2(relpos[1,:], relpos[0,:])  # Bearing
        
        # Add noise if requested
        if add_noise:
            noise = self.generate_noise(target_states.shape[1], rng, seed)
            Z = Z + noise
        
        # Wrap bearing to [0, 2π]
        Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        
        return Z

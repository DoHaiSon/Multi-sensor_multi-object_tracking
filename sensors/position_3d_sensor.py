import numpy as np
from .base_sensor import BaseSensor

class Position3DSensor(BaseSensor):
    """3D Position sensor implementation (x, y, z coordinates)."""
    
    @property
    def sensor_type(self):
        return 'pos_3D'
    
    @property
    def z_dim(self):
        return 3
    
    def _default_range_c(self):
        return np.array([[-4000, 4000], [-4000, 4000], [-1000, 1000]])
    
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """Generate 3D position measurements."""
        if target_states.size == 0:
            return np.array([])
        
        # Extract x, y, z positions
        Z = np.zeros((3, target_states.shape[1]))
        Z[0:3,:] = target_states[[0,2,4],:]
        
        # Add noise if requested
        if add_noise:
            noise = self.generate_noise(target_states.shape[1], rng, seed)
            Z = Z + noise
        
        return Z

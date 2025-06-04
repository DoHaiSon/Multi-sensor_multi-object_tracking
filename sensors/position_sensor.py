import numpy as np
from .base_sensor import BaseSensor

class PositionSensor(BaseSensor):
    """Position sensor implementation (x, y coordinates)."""
    
    @property
    def sensor_type(self):
        return 'pos'
    
    @property
    def z_dim(self):
        return 2
    
    def _default_range_c(self):
        return np.array([[-4000, 4000], [-4000, 4000]])
    
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """Generate position measurements."""
        if target_states.size == 0:
            return np.array([])
        
        # Direct position measurement
        Z = target_states[[0,2],:]  # x, y positions
        
        # Add noise if requested
        if add_noise:
            noise = self.generate_noise(target_states.shape[1], rng, seed)
            Z = Z + noise
        
        return Z

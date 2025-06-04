import numpy as np
from .base_sensor import BaseSensor

class RangeSensor(BaseSensor):
    """Range-only sensor implementation."""
    
    @property
    def sensor_type(self):
        return 'rng'
    
    @property
    def z_dim(self):
        return 1
    
    def _default_range_c(self):
        return np.array([0, 5000])
    
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """Generate range measurements."""
        if target_states.size == 0:
            return np.array([])
        
        # Calculate range from sensor to targets
        Z = np.sqrt((self.position[0] - target_states[0,:])**2 + 
                    (self.position[1] - target_states[2,:])**2)
        
        Z = Z.reshape(1, -1)  # Ensure 2D shape
        
        # Add noise if requested
        if add_noise:
            noise = self.generate_noise(target_states.shape[1], rng, seed)
            Z = Z + noise
        
        return Z

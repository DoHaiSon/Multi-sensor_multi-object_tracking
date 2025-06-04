import numpy as np
from .base_sensor import BaseSensor

class MilitaryRadarSensor(BaseSensor):
    """Military radar sensor (bearing, range, and range rate)."""
    
    @property
    def sensor_type(self):
        return 'brg_rng_rngrt'
    
    @property
    def z_dim(self):
        return 3
    
    def _default_range_c(self):
        return np.array([[0, 2*np.pi], [0, 5000], [-100, 100]])
    
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """Generate bearing, range, and range rate measurements."""
        if target_states.size == 0:
            return np.array([])
        
        relpos = target_states[[0,2],:] - self.position[:,None]
        relvel = target_states[[1,3],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        
        Z = np.zeros((3, target_states.shape[1]))
        Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])  # Bearing
        Z[1,:] = rng  # Range
        Z[2,:] = np.sum(relpos * relvel, axis=0) / rng  # Range rate
        
        # Add noise if requested
        if add_noise:
            noise = self.generate_noise(target_states.shape[1], rng, seed)
            Z = Z + noise
        
        # Wrap bearing to [0, 2π]
        Z[0,:] = np.mod(Z[0,:], 2*np.pi)
        
        return Z

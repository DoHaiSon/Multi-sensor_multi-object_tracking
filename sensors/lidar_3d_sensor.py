import numpy as np
from .base_sensor import BaseSensor

class Lidar3DSensor(BaseSensor):
    """3D LiDAR sensor (azimuth, elevation, range, range rate)."""
    
    @property
    def sensor_type(self):
        return 'az_el_rng'
    
    @property
    def z_dim(self):
        return 4
    
    def _default_range_c(self):
        return np.array([[-np.pi, np.pi], [-np.pi/2, np.pi/2], [0, 3000], [-50, 50]])
    
    def generate_measurement(self, target_states, add_noise=True, rng=None, seed=None):
        """Generate azimuth, elevation, range, and range rate measurements."""
        if target_states.size == 0:
            return np.array([])
        
        # Assuming 3D state: [x, vx, y, vy, z, vz]
        relpos = target_states[[0,2,4],:] - self.position[:,None]
        relvel = target_states[[1,3,5],:]
        rng = np.sqrt(np.sum(relpos**2, axis=0))
        
        Z = np.zeros((4, target_states.shape[1]))
        xy_rng = np.sqrt(np.sum(relpos[0:2,:]**2, axis=0))
        Z[0,:] = np.arctan2(relpos[0,:], relpos[1,:])  # Azimuth
        Z[1,:] = np.arctan2(xy_rng, relpos[2,:])  # Elevation
        Z[2,:] = rng  # Range
        Z[3,:] = np.sum(relpos * relvel, axis=0) / rng  # Range rate
        
        # Add noise if requested
        if add_noise:
            noise = self.generate_noise(target_states.shape[1], rng, seed)
            Z = Z + noise
        
        # Wrap angles to [-π, π]
        Z[0,:] = np.mod(Z[0,:] + np.pi, 2*np.pi) - np.pi
        Z[1,:] = np.mod(Z[1,:] + np.pi, 2*np.pi) - np.pi
        
        return Z

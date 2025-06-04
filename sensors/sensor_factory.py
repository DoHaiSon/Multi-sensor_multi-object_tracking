import numpy as np
from .bearing_sensor import BearingSensor
from .range_sensor import RangeSensor
from .bearing_range_sensor import BearingRangeSensor
from .position_sensor import PositionSensor
from .position_3d_sensor import Position3DSensor
from .doppler_sensor import DopplerSensor
from .military_radar_sensor import MilitaryRadarSensor
from .lidar_3d_sensor import Lidar3DSensor

class SensorFactory:
    """Factory class for creating sensors from configurations."""
    
    SENSOR_CLASS_MAP = {
        'brg': BearingSensor,
        'rng': RangeSensor,
        'brg_rng': BearingRangeSensor,
        'pos': PositionSensor,
        'pos_3D': Position3DSensor,
        'brg_rr': DopplerSensor,
        'brg_rng_rngrt': MilitaryRadarSensor,
        'az_el_rng': Lidar3DSensor,
    }
    
    @classmethod
    def create_sensor(cls, sensor_config):
        """
        Create a sensor from configuration dictionary.
        
        Args:
            sensor_config: Dictionary containing sensor configuration. Required keys:
                - type: Sensor type string
                - position: Sensor position [x, y] or [x, y, z]
                - P_D_rng: Detection probability range [P_D_first, P_D_second]
                - lambda_c_rng: Clutter rate range [lambda_c_first, lambda_c_second]
                - R: Measurement noise covariance matrix
                Optional keys:
                - id: Sensor identifier
                - range_c: Clutter range
        
        Returns:
            BaseSensor: Sensor instance of the specified type
        
        Raises:
            ValueError: If sensor type is not recognized
        """
        sensor_type = sensor_config['type']
        
        if sensor_type not in cls.SENSOR_CLASS_MAP:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        sensor_class = cls.SENSOR_CLASS_MAP[sensor_type]
        
        return sensor_class(
            sensor_id=sensor_config.get('id', 0),
            position=sensor_config['position'],
            P_D_rng=sensor_config['P_D_rng'],
            lambda_c_rng=sensor_config['lambda_c_rng'],
            R=sensor_config['R'],
            range_c=sensor_config.get('range_c')
        )
    
    @classmethod
    def get_available_sensor_types(cls):
        """
        Return list of available sensor types.
        
        Args:
            None
        
        Returns:
            list: List of available sensor type strings
        """
        return list(cls.SENSOR_CLASS_MAP.keys())

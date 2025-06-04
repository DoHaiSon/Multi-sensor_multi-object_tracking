import pytest
import numpy as np
import scipy.io
from config import get_args
from utils.common import select_model
from core.generate_truth import gen_truth
from core.generate_measurements import gen_measurements
from utils.matlab_rng import Matlab_RNG
from utils.logger import Logger

class Test_Generate_Measurements:
    """Test class for generate_measurements function with Brg_rng model"""
    
    @classmethod
    def setup_class(cls):
        """
        Setup test class - prepare model, truth data and MATLAB RNG.
        
        Args:
            None
        
        Returns:
            None: Initializes class variables for testing
        """
        # Initialize args with Brg_rng model
        cls.args = get_args([])
        cls.args.model = 'brg_rng' 
        cls.args.use_seed = True   
        cls.args.enable_logging = False
        cls.args.save_dataset = False  # Disable dataset saving for unit tests
        
        # Initialize logger
        cls.writer = Logger(enable_logging=False)
        
        # Initialize model
        cls.model = select_model(cls.args, cls.writer)
        
        # Initialize MATLAB-compatible RNG
        cls.rng = Matlab_RNG(seed=cls.args.seed)

        seed = cls.args.seed if cls.args.use_seed else None
        
        # Generate ground truth using MATLAB RNG
        cls.truth = gen_truth(cls.args, cls.model.dynamics, cls.rng, seed=seed)
        
        # Generate measurements using MATLAB RNG
        cls.python_meas = gen_measurements(cls.args, cls.model.sensors, cls.truth, cls.rng, seed=seed)
        
        # Load MATLAB measurements data for comparison
        try:
            matlab_meas = scipy.io.loadmat('tests/from_matlab/measurements_brg_rng.mat')
            cls.matlab_meas = matlab_meas['measurements']
        except FileNotFoundError:
            # If MATLAB reference file doesn't exist, skip MATLAB comparison tests
            cls.matlab_meas = None

    def test_measurements_structure(self):
        """
        Test measurements data structure.
        
        Args:
            None
        
        Returns:
            None: Asserts correct structure of measurements dictionary
        """
        # Check if all required fields exist
        assert 'K' in self.python_meas
        assert 'Z' in self.python_meas
        assert 'P_D' in self.python_meas
        assert 'lambda_c' in self.python_meas
        
        # Check dimensions
        assert len(self.python_meas['Z']) == self.args.K
        assert self.python_meas['P_D'].shape == (self.args.K, len(self.model.sensors))
        assert self.python_meas['lambda_c'].shape == (self.args.K, len(self.model.sensors))

    def test_measurements_P_D(self):
        """
        Test detection probability P_D.
        
        Args:
            None
        
        Returns:
            None: Asserts P_D values match MATLAB reference
        """
        if self.matlab_meas is None:
            pytest.skip("MATLAB reference data not available")
            
        assert np.allclose(
            self.matlab_meas['P_D'][0][0],
            self.python_meas['P_D'], 
            rtol=1e-6, atol=1e-8
        ), "Detection probabilities don't match"

    def test_measurements_lambda_c(self):
        """
        Test clutter rate lambda_c.
        
        Args:
            None
        
        Returns:
            None: Asserts lambda_c values match MATLAB reference
        """
        if self.matlab_meas is None:
            pytest.skip("MATLAB reference data not available")
            
        assert np.allclose(
            self.matlab_meas['lambda_c'][0][0],
            self.python_meas['lambda_c'], 
            rtol=1e-6, atol=1e-8
        ), "Clutter rates don't match"

    def test_measurements_Z(self):
        """
        Test measurements Z.
        
        Args:
            None
        
        Returns:
            None: Asserts measurement values match MATLAB reference
        """
        if self.matlab_meas is None:
            pytest.skip("MATLAB reference data not available")
            
        for k in range(self.args.K):
            for s in range(len(self.model.sensors)):
                matlab_Z = self.matlab_meas['Z'][0][0][k][s]
                python_Z = self.python_meas['Z'][k][s]
                
                # First check if both are empty or both have data
                if matlab_Z.size == 0 and python_Z.size == 0:
                    continue
                    
                # If both have data, compare values with detailed error reporting
                if matlab_Z.size > 0 and python_Z.size > 0:
                    # Check if shapes match first
                    if matlab_Z.shape != python_Z.shape:
                        print(f"Shape mismatch at time {k}, sensor {s}:")
                        print(f"MATLAB shape: {matlab_Z.shape}, Python shape: {python_Z.shape}")
                        assert False, f"Shape mismatch at time {k}, sensor {s}"
                    
                    # Check element-wise differences
                    if not np.allclose(matlab_Z, python_Z, rtol=1e-6, atol=1e-8):
                        diff = np.abs(matlab_Z - python_Z)
                        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                        max_diff = diff[max_diff_idx]
                        
                        print(f"Measurement mismatch at time {k}, sensor {s}:")
                        print(f"Maximum difference: {max_diff} at position {max_diff_idx}")
                        print(f"MATLAB value at {max_diff_idx}: {matlab_Z[max_diff_idx]}")
                        print(f"Python value at {max_diff_idx}: {python_Z[max_diff_idx]}")
                        print(f"Relative difference: {max_diff / abs(matlab_Z[max_diff_idx]) if matlab_Z[max_diff_idx] != 0 else 'inf'}")
                        
                        # Show first few differences for debugging
                        print(f"First 5 MATLAB values: {matlab_Z.flatten()[:5]}")
                        print(f"First 5 Python values: {python_Z.flatten()[:5]}")
                        print(f"First 5 differences: {diff.flatten()[:5]}")
                        
                        assert False, f"Measurements don't match at time {k}, sensor {s}"

    def test_measurements_consistency(self):
        """
        Test consistency of measurements with model parameters.
        
        Args:
            None
        
        Returns:
            None: Asserts measurements are consistent with sensor configurations
        """
        for k in range(self.args.K):
            for s in range(len(self.model.sensors)):
                if isinstance(self.python_meas['Z'][k][s], np.ndarray) and self.python_meas['Z'][k][s].size > 0:
                    # Check measurement dimension - use sensor object properties
                    sensor = self.model.sensors[s]
                    expected_z_dim = getattr(sensor, 'z_dim', getattr(sensor, 'measurement_dim', 2))
                    assert self.python_meas['Z'][k][s].shape[0] == expected_z_dim, \
                        f"Wrong measurement dimension at time {k}, sensor {s}"

    def test_sensor_object_structure(self):
        """
        Test that sensors are properly created as objects.
        
        Args:
            None
        
        Returns:
            None: Asserts sensors are proper objects with required methods
        """
        assert len(self.model.sensors) > 0, "No sensors found"
        
        for i, sensor in enumerate(self.model.sensors):
            # Check that sensor has required attributes/methods
            assert hasattr(sensor, 'generate_measurement'), f"Sensor {i} missing generate_measurement method"
            assert hasattr(sensor, 'position'), f"Sensor {i} missing position attribute"
            assert hasattr(sensor, 'sensor_type'), f"Sensor {i} missing sensor_type attribute"

    @classmethod
    def teardown_class(cls):
        """
        Cleanup after tests.
        
        Args:
            None
        
        Returns:
            None: Cleans up class resources
        """
        if hasattr(cls, 'rng'):
            del cls.rng
        if hasattr(cls, 'model'):
            del cls.model
        if hasattr(cls, 'truth'):
            del cls.truth
        if hasattr(cls, 'writer'):
            del cls.writer

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
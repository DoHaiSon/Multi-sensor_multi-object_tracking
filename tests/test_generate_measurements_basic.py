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
    """Test class for generate_measurements function with Basic model"""
    
    @classmethod
    def setup_class(cls):
        """Setup test class - prepare model, truth data and MATLAB RNG"""
        # Initialize args with Basic model
        cls.args = get_args([])
        cls.args.model = 'Basic'    
        cls.args.z_dim = 2
        cls.args.enable_logging = False
        
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
        if cls.args.model == 'Basic':
            matlab_meas = scipy.io.loadmat('tests/from_matlab/measurements_basic.mat')
        elif cls.args.model == 'Brg':
            matlab_meas = scipy.io.loadmat('tests/from_matlab/measurements_brg.mat')
        elif cls.args.model == 'Brg_rng':
            matlab_meas = scipy.io.loadmat('tests/from_matlab/measurements_brg_rng.mat')
        else:
            raise ValueError(f"Model '{cls.args.model}' not supported for testing.")
        cls.matlab_meas = matlab_meas['measurements']

    def test_measurements_structure(self):
        """Test measurements data structure"""
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
        """Test detection probability P_D"""
        assert np.allclose(
            self.matlab_meas['P_D'][0][0],  # Remove extra dimensions
            self.python_meas['P_D'], 
            rtol=1e-10
        ), "Detection probabilities don't match"

    def test_measurements_lambda_c(self):
        """Test clutter rate lambda_c"""
        assert np.allclose(
            self.matlab_meas['lambda_c'][0][0],  # Remove extra dimensions
            self.python_meas['lambda_c'], 
            rtol=1e-10
        ), "Clutter rates don't match"

    def test_measurements_Z(self):
        """Test measurements Z"""
        for k in range(self.args.K):
            for s in range(len(self.model.sensors)):
                matlab_Z = self.matlab_meas['Z'][0][0][k][s]
                python_Z = self.python_meas['Z'][k][s]
                
                # First check if both are empty or both have data
                if matlab_Z.size == 0 and python_Z.size == 0:
                    continue
                    
                # If both have data, compare values
                assert np.allclose(
                    matlab_Z, 
                    python_Z, 
                    rtol=1e-10
                ), f"Measurements don't match at time {k}, sensor {s}"

    def test_measurements_consistency(self):
        """Test consistency of measurements with model parameters"""
        for k in range(self.args.K):
            for s in range(len(self.model.sensors)):
                if isinstance(self.python_meas['Z'][k][s], np.ndarray) and self.python_meas['Z'][k][s].size > 0:
                    # Check measurement dimension
                    assert self.python_meas['Z'][k][s].shape[0] == self.model.sensors[s]['z_dim'], \
                        f"Wrong measurement dimension at time {k}, sensor {s}"

    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
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
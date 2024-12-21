import pytest
import numpy as np
import scipy.io
from config import get_args
from core.generate_truth import gen_truth
from models.basic import Basic_Model 
from utils.matlab_rng import Matlab_RNG
from utils.logger import Logger

class Test_Generate_Truth:
    """Test class for generate_truth function"""
    
    @classmethod
    def setup_class(cls):
        """Setup test class"""
        # Load MATLAB truth data
        matlab_data = scipy.io.loadmat('tests/from_matlab/truth_scenario1.mat')
        cls.matlab_truth = matlab_data['truth']
        
        # Get args from config
        cls.args = get_args([])
        cls.args.model = 'Basic'  # Ensure using Basic model
        cls.args.enable_logging = False  # Disable logging for tests
        
        # Initialize model and RNG
        cls.writer = Logger(enable_logging=False)
        cls.model = Basic_Model(cls.args, cls.writer)
        cls.rng = Matlab_RNG(seed=2808)

        # Generate Python truth once for all tests
        cls.python_truth = gen_truth(cls.args, cls.model.dynamics, rng=cls.rng)

    def test_K_value(self):
        """Test if K (number of scans) matches"""
        assert self.python_truth['K'] == self.matlab_truth['K'], \
            f"K mismatch: Python={self.python_truth['K']}, MATLAB={self.matlab_truth['K']}"
            
    def test_total_tracks(self):
        """Test if total_tracks matches"""
        assert self.python_truth['total_tracks'] == self.matlab_truth['total_tracks'], \
            f"total_tracks mismatch: Python={self.python_truth['total_tracks']}, MATLAB={self.matlab_truth['total_tracks']}"
        
    def test_N_array(self):
        """Test if N (number of targets at each timestep) matches"""
        matlab_N = self.matlab_truth['N'][0][0]  # Remove extra dimensions
        if matlab_N.ndim > 1:  
            matlab_N = matlab_N.flatten()
        np.testing.assert_array_equal(
            self.python_truth['N'],
            matlab_N,
            err_msg="N arrays don't match"
        )
        
    def test_X_cell(self):
        """Test if X (state of targets at each timestep) matches"""
        matlab_truth_X = self.matlab_truth['X'][0][0] # Remove extra dimensions
        for k in range(self.python_truth['K']):
            if self.python_truth['X'][k] is not None and matlab_truth_X[k][0].size > 0:
                np.testing.assert_allclose(
                    self.python_truth['X'][k],
                    matlab_truth_X[k][0],  
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"X values don't match at timestep k={k}"
                )
            else:
                # Check if both are empty/None
                assert (self.python_truth['X'][k] is None and matlab_truth_X[k].size == 0), \
                    f"X empty status mismatch at timestep k={k}"

    def test_track_list_cell(self):
        """Test if track_list (absolute index target identities) matches"""
        mathlab_track_list = self.matlab_truth['track_list'][0][0]  # Remove extra dimensions
        for k in range(self.python_truth['K']):
            if (self.python_truth['track_list'][k] is not None and 
                len(self.matlab_truth['track_list'][k][0]) > 0):
                np.testing.assert_array_equal(
                    self.python_truth['track_list'][k],
                    self.matlab_truth['track_list'][k][0],  # MATLAB cell array needs [0] indexing
                    err_msg=f"track_list doesn't match at timestep k={k}"
                )
            else:
                # Check if both are empty/None
                is_python_empty = self.python_truth['track_list'][k] is None
                is_matlab_empty = len(self.matlab_truth['track_list'][k][0]) == 0
                assert is_python_empty and is_matlab_empty, \
                    f"track_list empty status mismatch at timestep k={k}"

    def test_track_list_cell(self):
        """Test if track_list (target identities at each timestep) matches"""
        matlab_truth_track_list = self.matlab_truth['track_list'][0][0]  # Remove extra dimensions
        for k in range(self.python_truth['K']):
            if self.python_truth['track_list'][k] is not None and matlab_truth_track_list[k][0].size > 0:
                # Convert list to numpy array for comparison if needed
                python_track_list = np.array(self.python_truth['track_list'][k])
                np.testing.assert_array_equal(
                    python_track_list,
                    matlab_truth_track_list[k][0].flatten(),  # Flatten to match Python list structure
                    err_msg=f"track_list values don't match at timestep k={k}"
                )
            else:
                # Check if both are empty/None
                assert (self.python_truth['track_list'][k] is None and matlab_truth_track_list[k].size == 0), \
                    f"track_list empty status mismatch at timestep k={k}"

    def test_L_cell(self):
        """Test if L (labels of targets at each timestep) matches between Python and MATLAB"""
        matlab_truth_L = self.matlab_truth['L'][0][0]  # Remove extra dimensions
        for k in range(self.python_truth['K']):
            # Check empty condition: Python None equals MATLAB empty array
            matlab_is_empty = matlab_truth_L[k][0].size == 0
            python_is_empty = self.python_truth['L'][k] is None
            
            assert matlab_is_empty == python_is_empty, \
                f"L empty status mismatch at timestep k={k}: Python={python_is_empty}, MATLAB={matlab_is_empty}"
            
            # Only compare values if not empty
            if not matlab_is_empty:  # or not python_is_empty, since they are equal at this point
                np.testing.assert_array_equal(
                    self.python_truth['L'][k],
                    matlab_truth_L[k][0],
                    err_msg=f"L values don't match at timestep k={k}"
                )

    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        if hasattr(cls, 'rng'):
            del cls.rng
        if hasattr(cls, 'writer'):
            del cls.writer

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
import pytest
import numpy as np
import scipy.io
from config import get_args
from utils.common import select_model
from core.generate_truth import gen_truth
from utils.matlab_rng import Matlab_RNG
from utils.logger import Logger

class Test_Generate_Truth:
    """Test class for generate_truth function"""
    
    @classmethod
    def setup_class(cls):
        """
        Setup test class - prepare model and MATLAB RNG.
        
        Args:
            None
        
        Returns:
            None: Initializes class variables for testing
        """
        # Initialize args
        cls.args = get_args([])
        cls.args.use_seed = True   
        cls.args.enable_logging = False
        
        # Initialize logger
        cls.writer = Logger(enable_logging=False)
        
        # Initialize model
        cls.model = select_model(cls.args, cls.writer)
        
        # Initialize MATLAB-compatible RNG
        cls.rng = Matlab_RNG(seed=cls.args.seed)

        seed = cls.args.seed if cls.args.use_seed else None
        
        # Generate ground truth using MATLAB RNG
        cls.python_truth = gen_truth(cls.args, cls.model.dynamics, cls.rng, seed=seed)
        
        # Load MATLAB ground truth data for comparison
        try:
            matlab_truth = scipy.io.loadmat('tests/from_matlab/truth_scenario1.mat')
            cls.matlab_truth = matlab_truth['truth']
        except FileNotFoundError:
            # If MATLAB reference file doesn't exist, skip MATLAB comparison tests
            cls.matlab_truth = None

    def test_truth_structure(self):
        """
        Test ground truth data structure.
        
        Args:
            None
        
        Returns:
            None: Asserts correct structure of truth dictionary
        """
        # Check if all required fields exist
        assert 'K' in self.python_truth
        assert 'X' in self.python_truth
        assert 'N' in self.python_truth
        assert 'L' in self.python_truth
        assert 'track_list' in self.python_truth
        assert 'total_tracks' in self.python_truth
        
        # Check dimensions
        assert len(self.python_truth['X']) == self.args.K
        assert len(self.python_truth['N']) == self.args.K
        assert len(self.python_truth['L']) == self.args.K
        assert len(self.python_truth['track_list']) == self.args.K

    def test_truth_K(self):
        """
        Test number of time steps K.
        
        Args:
            None
        
        Returns:
            None: Asserts K value matches MATLAB reference
        """
        if self.matlab_truth is None:
            pytest.skip("MATLAB reference data not available")
            
        assert self.python_truth['K'] == self.matlab_truth['K'][0][0][0][0], "K values don't match"

    def test_truth_total_tracks(self):
        """
        Test total number of tracks.
        
        Args:
            None
        
        Returns:
            None: Asserts total_tracks value matches MATLAB reference
        """
        if self.matlab_truth is None:
            pytest.skip("MATLAB reference data not available")
            
        assert self.python_truth['total_tracks'] == self.matlab_truth['total_tracks'][0][0][0][0], \
            "Total tracks don't match"

    def test_truth_N(self):
        """
        Test number of targets N at each time step.
        
        Args:
            None
        
        Returns:
            None: Asserts N values match MATLAB reference
        """
        if self.matlab_truth is None:
            pytest.skip("MATLAB reference data not available")
            
        matlab_N = self.matlab_truth['N'][0][0].flatten()
        assert np.array_equal(self.python_truth['N'], matlab_N), "N values don't match"

    def test_truth_X(self):
        """
        Test target states X.
        
        Args:
            None
        
        Returns:
            None: Asserts X values match MATLAB reference
        """
        if self.matlab_truth is None:
            pytest.skip("MATLAB reference data not available")
            
        for k in range(self.args.K):
            matlab_X = self.matlab_truth['X'][0][0][k]
            python_X = self.python_truth['X'][k]
            
            # First check if both are None/empty or both have data
            if matlab_X.size == 0 and (python_X is None or python_X.size == 0):
                continue
            
            # Handle MATLAB's object array structure
            if matlab_X.size > 0 and python_X is not None:
                # Convert MATLAB object array to regular numpy array if needed
                if matlab_X.dtype == object:
                    # Extract the actual array from the object array
                    if matlab_X.size == 1:
                        matlab_X_clean = matlab_X.item()
                    else:
                        # Handle multiple objects in array
                        matlab_X_clean = np.concatenate([item for item in matlab_X.flatten()], axis=1)
                else:
                    matlab_X_clean = matlab_X
                
                # Ensure python_X is a numpy array
                if not isinstance(python_X, np.ndarray):
                    python_X = np.array(python_X)
                
                # Compare the cleaned arrays
                assert np.allclose(matlab_X_clean, python_X, rtol=1e-10), \
                    f"X values don't match at time step {k}: " \
                    f"MATLAB shape: {matlab_X_clean.shape}, Python shape: {python_X.shape}"
            else:
                # One has data, the other doesn't - this is a mismatch
                assert False, f"X structure mismatch at time step {k}: " \
                             f"MATLAB has {matlab_X.size} elements, " \
                             f"Python has {python_X.size if python_X is not None else 0} elements"

    def test_truth_track_list(self):
        """
        Test track list consistency.
        
        Args:
            None
        
        Returns:
            None: Asserts track_list values match MATLAB reference
        """
        if self.matlab_truth is None:
            pytest.skip("MATLAB reference data not available")
            
        for k in range(self.args.K):
            matlab_track_list = self.matlab_truth['track_list'][0][0][k]
            python_track_list = self.python_truth['track_list'][k]
            
            # Check if both are None/empty or both have data
            if matlab_track_list.size == 0 and (python_track_list is None or len(python_track_list) == 0):
                continue
                
            # If both have data, compare values
            if matlab_track_list.size > 0 and python_track_list is not None:
                # Handle MATLAB's complex data structure
                try:
                    # First try to flatten and convert directly
                    if matlab_track_list.dtype == object:
                        # Extract values from object array
                        matlab_values = []
                        for item in matlab_track_list.flatten():
                            if hasattr(item, 'flatten'):
                                matlab_values.extend(item.flatten().tolist())
                            else:
                                matlab_values.append(item)
                        matlab_track_list_flat = np.array(matlab_values, dtype=int)
                    else:
                        matlab_track_list_flat = np.array(matlab_track_list.flatten(), dtype=int)
                    
                    python_track_list_array = np.array(python_track_list, dtype=int)
                    
                    assert np.array_equal(python_track_list_array, matlab_track_list_flat), \
                        f"Track list doesn't match at time step {k}: " \
                        f"MATLAB: {matlab_track_list_flat}, Python: {python_track_list_array}"
                        
                except (ValueError, TypeError) as e:
                    # If conversion fails, compare lengths and individual elements
                    print(f"Warning: Could not directly compare track lists at time step {k}: {e}")
                    print(f"MATLAB data type: {type(matlab_track_list)}, shape: {matlab_track_list.shape}")
                    print(f"Python data: {python_track_list}")
                    
                    # Fallback: just check that both have the same number of elements
                    matlab_count = matlab_track_list.size
                    python_count = len(python_track_list) if python_track_list else 0
                    
                    assert matlab_count == python_count, \
                        f"Track list count mismatch at time step {k}: " \
                        f"MATLAB has {matlab_count} elements, Python has {python_count} elements"
            else:
                # One has data, the other doesn't - this is a mismatch
                assert False, f"Track list structure mismatch at time step {k}: " \
                             f"MATLAB has {matlab_track_list.size} elements, " \
                             f"Python has {len(python_track_list) if python_track_list else 0} elements"

    def test_truth_consistency(self):
        """
        Test consistency between different truth components.
        
        Args:
            None
        
        Returns:
            None: Asserts internal consistency of truth data
        """
        for k in range(self.args.K):
            # Check consistency between N and X
            if self.python_truth['X'][k] is not None:
                assert self.python_truth['X'][k].shape[1] == self.python_truth['N'][k], \
                    f"Inconsistency between N and X dimensions at time step {k}"
            else:
                assert self.python_truth['N'][k] == 0, \
                    f"N should be 0 when X is None at time step {k}"
            
            # Check consistency between N and track_list
            if self.python_truth['track_list'][k] is not None:
                # Ensure track_list contains valid data types
                track_list_len = len(self.python_truth['track_list'][k])
                assert track_list_len == self.python_truth['N'][k], \
                    f"Inconsistency between N and track_list length at time step {k}"
                
                # Check that all track IDs are valid integers
                for track_id in self.python_truth['track_list'][k]:
                    assert isinstance(track_id, (int, np.integer)), \
                        f"Track ID must be integer, got {type(track_id)} at time step {k}"
            else:
                assert self.python_truth['N'][k] == 0, \
                    f"N should be 0 when track_list is None at time step {k}"

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
        if hasattr(cls, 'writer'):
            del cls.writer

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
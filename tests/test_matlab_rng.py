import pytest
import numpy as np
import scipy.io
from utils.matlab_rng import Matlab_RNG

class Test_Matlab_RNG:
    @classmethod
    def setup_class(cls):
        """Setup test class - load MATLAB test data"""
        cls.matlab_data = scipy.io.loadmat('tests/from_matlab/matlab_rng_test_data.mat')['test_data'][0,0]

    def setup_method(self):
        """Setup before each test method - initialize RNG with seed"""
        self.rng = Matlab_RNG(seed=2808)

    def test_randn(self):
        """Test randn with various input combinations"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test single value (no args)
        val = self.rng.randn()
        np.testing.assert_almost_equal(val, self.matlab_data['randn_1'][0,0])
        
        # Test single argument -> (n,1) vector
        val = self.rng.randn(1)
        np.testing.assert_array_almost_equal(val, self.matlab_data['randn_2'])
        
        # Test matrix (m,n)
        val = self.rng.randn(3, 2)
        np.testing.assert_array_almost_equal(val, self.matlab_data['randn_3'])
        
        # Test column vector
        val = self.rng.randn(5, 1)
        np.testing.assert_array_almost_equal(val, self.matlab_data['randn_4'])
        
        # Test row vector
        val = self.rng.randn(1, 4)
        np.testing.assert_array_almost_equal(val, self.matlab_data['randn_5'])

    def test_rand(self):
        """Test rand with various input combinations"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test single value (no args)
        val = self.rng.rand()
        np.testing.assert_almost_equal(val, self.matlab_data['rand_1'][0,0])
        
        # Test single argument -> (n,1) vector
        val = self.rng.rand(3)
        np.testing.assert_array_almost_equal(val, self.matlab_data['rand_2'])
        
        # Test matrix (m,n)
        val = self.rng.rand(2, 4)
        np.testing.assert_array_almost_equal(val, self.matlab_data['rand_3'])
        
        # Test column vector
        val = self.rng.rand(4, 1)
        np.testing.assert_array_almost_equal(val, self.matlab_data['rand_4'])
        
        # Test row vector
        val = self.rng.rand(1, 3)
        np.testing.assert_array_almost_equal(val, self.matlab_data['rand_5'])

    def test_poisson(self):
        """Test poisson with various input combinations"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test single value
        val = self.rng.poisson(5)
        np.testing.assert_equal(val, self.matlab_data['poisson_1'][0,0])
        
        # Test column vector
        val = self.rng.poisson(10, 3, 1)
        np.testing.assert_array_equal(val, self.matlab_data['poisson_2'])
        
        # Test row vector
        val = self.rng.poisson(15, 1, 4)
        np.testing.assert_array_equal(val, self.matlab_data['poisson_3'])
        
        # Test matrix
        val = self.rng.poisson(7, 2, 3)
        np.testing.assert_array_equal(val, self.matlab_data['poisson_4'])

    def test_normal(self):
        """Test normal with various input combinations"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test single value
        val = self.rng.normal(0, 1)
        np.testing.assert_almost_equal(val, self.matlab_data['normal_1'][0,0])
        
        # Test single argument -> (n,1) vector
        val = self.rng.normal(2, 0.5, 4)
        np.testing.assert_array_almost_equal(val, self.matlab_data['normal_2'])
        
        # Test matrix (m,n)
        val = self.rng.normal(2, 0.5, 3, 2)
        np.testing.assert_array_almost_equal(val, self.matlab_data['normal_3'])
        
        # Test column vector
        val = self.rng.normal(-1, 2, 5, 1)
        np.testing.assert_array_almost_equal(val, self.matlab_data['normal_4'])
        
        # Test row vector
        val = self.rng.normal(1, 0.1, 1, 3)
        np.testing.assert_array_almost_equal(val, self.matlab_data['normal_5'])

    def test_uniform(self):
        """Test uniform with various input combinations"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test single value
        val = self.rng.uniform(-1, 1)
        np.testing.assert_almost_equal(val, self.matlab_data['uniform_1'][0,0])
        
        # Test single argument -> (n,1) vector
        val = self.rng.uniform(0, 5, 3)
        np.testing.assert_array_almost_equal(val, self.matlab_data['uniform_2'])
        
        # Test matrix (m,n)
        val = self.rng.uniform(0, 5, 2, 3)
        np.testing.assert_array_almost_equal(val, self.matlab_data['uniform_3'])
        
        # Test column vector
        val = self.rng.uniform(-2, 2, 4, 1)
        np.testing.assert_array_almost_equal(val, self.matlab_data['uniform_4'])
        
        # Test row vector
        val = self.rng.uniform(0, 1, 1, 5)
        np.testing.assert_array_almost_equal(val, self.matlab_data['uniform_5'])

    def test_randi(self):
        """Test randi with various input combinations"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test single value
        val = self.rng.randi(10)
        np.testing.assert_equal(val, self.matlab_data['randi_1'][0,0])
        
        # Test single argument -> (n,1) vector
        val = self.rng.randi(10, 5)
        np.testing.assert_array_equal(val, self.matlab_data['randi_2'])
        
        # Test matrix (m,n)
        val = self.rng.randi(100, 3, 3)
        np.testing.assert_array_equal(val, self.matlab_data['randi_3'])
        
        # Test column vector
        val = self.rng.randi(50, 4, 1)
        np.testing.assert_array_equal(val, self.matlab_data['randi_4'])
        
        # Test row vector
        val = self.rng.randi(20, 1, 6)
        np.testing.assert_array_equal(val, self.matlab_data['randi_5'])

    def test_randperm(self):
        """Test randperm with various input sizes"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test small permutation
        val = self.rng.randperm(5)
        np.testing.assert_array_equal(val, self.matlab_data['randperm_1'][0])
        
        # Test medium permutation
        val = self.rng.randperm(10)
        np.testing.assert_array_equal(val, self.matlab_data['randperm_2'][0])
        
        # Test large permutation
        val = self.rng.randperm(15)
        np.testing.assert_array_equal(val, self.matlab_data['randperm_3'][0])

    def test_multivariate_normal(self):
        """Test multivariate normal with various input combinations"""
        # Reset RNG for this specific test
        self.rng = Matlab_RNG(seed=2808)
        
        # Test single sample, 2D standard normal
        val = self.rng.multivariate_normal(np.zeros(2), np.eye(2))
        np.testing.assert_array_almost_equal(val, self.matlab_data['mvnorm_1'].flatten())
        
        # Test single sample, 2D with correlation
        mean = np.array([1, 2])
        cov = np.array([[2, 0.5], [0.5, 1]])
        val = self.rng.multivariate_normal(mean, cov)
        np.testing.assert_array_almost_equal(val, self.matlab_data['mvnorm_2'].flatten())
        
        # Test multiple samples, 2D standard normal
        val = self.rng.multivariate_normal(np.zeros(2), np.eye(2), size=3)
        np.testing.assert_array_almost_equal(val, self.matlab_data['mvnorm_3'])
        
        # Test multiple samples, 3D
        mean = np.array([1, 2, 3])
        val = self.rng.multivariate_normal(mean, np.eye(3), size=4)
        np.testing.assert_array_almost_equal(val, self.matlab_data['mvnorm_4'])
        
        # Test multiple samples, 1D
        mean = np.array([0])
        cov = np.array([[1]])
        val = self.rng.multivariate_normal(mean, cov, size=5)
        np.testing.assert_array_almost_equal(val, self.matlab_data['mvnorm_5'])

        # Additional dimension tests
        # Test output shape when size is not specified (should be 1D array)
        val = self.rng.multivariate_normal(np.zeros(3), np.eye(3))
        assert val.shape == (3,)

        # Test output shape with size parameter
        val = self.rng.multivariate_normal(np.zeros(2), np.eye(2), size=4)
        assert val.shape == (4, 2)

    def test_multivariate_normal_errors(self):
        """Test error handling for invalid inputs in multivariate_normal"""
        # Test non-matching dimensions between mean and covariance
        with pytest.raises(ValueError):
            self.rng.multivariate_normal(np.zeros(3), np.eye(2))

        # Test non-square covariance matrix
        with pytest.raises(ValueError):
            self.rng.multivariate_normal(np.zeros(2), np.array([[1, 0], [0, 1], [0, 0]]))

        # Test 1D input for covariance (must be 2D)
        with pytest.raises(ValueError):
            self.rng.multivariate_normal(np.zeros(2), np.array([1, 1]))

        # Test invalid size parameter
        with pytest.raises(ValueError):
            self.rng.multivariate_normal(np.zeros(2), np.eye(2), size=-1)
            
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        with pytest.raises(Exception):
            self.rng.rand(-1, 2)
        with pytest.raises(Exception):
            self.rng.randn(1.5, 2)
        with pytest.raises(Exception):
            self.rng.poisson(-5)
        with pytest.raises(Exception):
            self.rng.normal(0, -1)
        with pytest.raises(Exception):
            self.rng.uniform(2, 1)
        with pytest.raises(Exception):
            self.rng.randi(-10)
        with pytest.raises(Exception):
            self.rng.randperm(0)

    def test_dimension_consistency(self):
        """Test that output dimensions match input specifications"""
        assert np.isscalar(self.rng.rand())
        assert np.isscalar(self.rng.randn())
        assert np.isscalar(self.rng.poisson(5))
        assert np.isscalar(self.rng.normal(0, 1))
        assert np.isscalar(self.rng.uniform(0, 1))
        assert np.isscalar(self.rng.randi(10))

        assert self.rng.rand(5).shape == (5, 5)
        assert self.rng.randn(3).shape == (3, 3)
        assert self.rng.normal(0, 1, 4).shape == (4, 4)
        assert self.rng.uniform(0, 1, 6).shape == (6, 6)
        assert self.rng.randi(10, 7).shape == (7, 7)

        assert self.rng.rand(2, 3).shape == (2, 3)
        assert self.rng.randn(4, 2).shape == (4, 2)
        assert self.rng.normal(0, 1, 3, 2).shape == (3, 2)
        assert self.rng.uniform(0, 1, 2, 4).shape == (2, 4)
        assert self.rng.randi(10, 3, 3).shape == (3, 3)

    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'rng'):
            del self.rng

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
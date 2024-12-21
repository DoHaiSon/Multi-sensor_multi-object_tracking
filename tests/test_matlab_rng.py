import os
import pytest
import numpy as np
import scipy.io
from utils.matlab_rng import Matlab_RNG

class Test_Matlab_RNG:
    @classmethod
    def setup_class(cls):
        """Setup test class by loading MATLAB test data"""
        # Load MATLAB generated test data
        matlab_data = scipy.io.loadmat('tests/from_matlab/matlab_rng_test_data.mat')
        cls.test_data = matlab_data['test_data'][0][0]
        
        # Initialize MatlabRNG with same seed
        cls.rng = Matlab_RNG(seed=2808)

    def test_randn(self):
        """Test standard normal random numbers"""
        # Test single value
        result = self.rng.randn(1, 1)
        expected = self.test_data['randn_1']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

        # Test matrix
        result = self.rng.randn(3, 2)
        expected = self.test_data['randn_2']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

        # Test column vector
        result = self.rng.randn(5, 1)
        expected = self.test_data['randn_3']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_rand(self):
        """Test uniform random numbers"""
        # Test single value
        result = self.rng.rand(1, 1)
        expected = self.test_data['rand_1']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

        # Test matrix
        result = self.rng.rand(2, 4)
        expected = self.test_data['rand_2']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

        # Test column vector
        result = self.rng.rand(4, 1)
        expected = self.test_data['rand_3']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_poisson(self):
        """Test Poisson random numbers"""
        lambda_vals = [5, 10, 15]
        for i, lam in enumerate(lambda_vals):
            result = self.rng.poisson(lam)
            expected = self.test_data['poisson'][0][i]
            assert result == expected, f"Poisson test failed for lambda={lam}"

    def test_normal(self):
        """Test normal distribution"""
        # Test single value
        result = self.rng.normal(0, 1)
        expected = self.test_data['normal_1'][0][0]
        np.testing.assert_allclose(result, expected, rtol=1e-15)

        # Test matrix with different mean and std
        result = self.rng.normal(2, 0.5, size=(3, 2))
        expected = self.test_data['normal_2']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_uniform(self):
        """Test uniform distribution"""
        # Test single value
        result = self.rng.uniform(-1, 1)
        expected = self.test_data['uniform_1'][0][0]
        np.testing.assert_allclose(result, expected, rtol=1e-15)

        # Test matrix
        result = self.rng.uniform(0, 5, size=(2, 3))
        expected = self.test_data['uniform_2']
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_randi(self):
        """Test random integers"""
        # Test single value
        result = self.rng.randi(10, 1, 1)
        expected = self.test_data['randi_1']
        np.testing.assert_array_equal(result, expected)

        # Test matrix
        result = self.rng.randi(100, 3, 3)
        expected = self.test_data['randi_2']
        np.testing.assert_array_equal(result, expected)

    def test_randperm(self):
        """Test random permutation"""
        result = self.rng.randperm(10)
        expected = self.test_data['randperm']
        np.testing.assert_array_equal(result, expected)

    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        if hasattr(cls, 'rng'):
            del cls.rng

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
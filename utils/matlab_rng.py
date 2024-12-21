import numpy as np
import matlab.engine

class Matlab_RNG:
    """
    Wrapper class for MATLAB random number generator in Python.
    This class ensures random numbers match MATLAB's output exactly.
    """
    def __init__(self, seed=None):
        """
        Initialize MATLAB engine and set random seed.

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.eng = matlab.engine.start_matlab()
        if seed is not None:
            self.eng.eval(f'rng({seed})', nargout=0)

    def randn(self, *args):
        """
        Generate standard normal random numbers using MATLAB's randn.
        Equivalent to np.random.randn()

        Parameters:
        -----------
        *args : tuple of int
            Shape of output array. Can be (m,n) or m,n

        Returns:
        --------
        np.ndarray
            Array of random numbers with specified shape
        """
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            m, n = args[0]
        else:
            m, n = args
        return np.array(self.eng.randn(float(m), float(n)))

    def rand(self, *args):
        """
        Generate uniformly distributed random numbers using MATLAB's rand.
        Equivalent to np.random.rand()

        Parameters:
        -----------
        *args : tuple of int
            Shape of output array. Can be (m,n) or m,n

        Returns:
        --------
        np.ndarray
            Array of random numbers with specified shape
        """
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            m, n = args[0]
        else:
            m, n = args
        return np.array(self.eng.rand(float(m), float(n)))

    def poisson(self, lam):
        """
        Generate random numbers from Poisson distribution using MATLAB's poissrnd.
        Equivalent to np.random.poisson()

        Parameters:
        -----------
        lam : float
            The expected number of events occurring (lambda parameter)

        Returns:
        --------
        int
            Random number from Poisson distribution
        """
        return int(self.eng.poissrnd(float(lam)))

    def normal(self, mu, sigma, size=None):
        """
        Generate random numbers from normal distribution using MATLAB's normrnd.
        Equivalent to np.random.normal()

        Parameters:
        -----------
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution
        size : tuple, optional
            Output shape. Default is None (single value)

        Returns:
        --------
        np.ndarray or float
            Random samples from normal distribution
        """
        if size is None:
            return float(self.eng.normrnd(float(mu), float(sigma)))
        m, n = size
        return np.array(self.eng.normrnd(float(mu), float(sigma), float(m), float(n)))

    def uniform(self, low, high, size=None):
        """
        Generate random numbers from uniform distribution using MATLAB's unifrnd.
        Equivalent to np.random.uniform()

        Parameters:
        -----------
        low : float
            Lower boundary of output interval
        high : float
            Upper boundary of output interval
        size : tuple, optional
            Output shape. Default is None (single value)

        Returns:
        --------
        np.ndarray or float
            Random samples from uniform distribution
        """
        if size is None:
            return float(self.eng.unifrnd(float(low), float(high)))
        m, n = size
        return np.array(self.eng.unifrnd(float(low), float(high), float(m), float(n)))

    def randi(self, imax, *args):
        """
        Generate random integers using MATLAB's randi.
        Equivalent to np.random.randint()

        Parameters:
        -----------
        imax : int
            Upper bound of random integers (inclusive)
        *args : tuple of int
            Shape of output array. Can be (m,n) or m,n

        Returns:
        --------
        np.ndarray
            Array of random integers with specified shape
        """
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            m, n = args[0]
        else:
            m, n = args
        return np.array(self.eng.randi(float(imax), float(m), float(n)))

    def randperm(self, n):
        """
        Generate random permutation using MATLAB's randperm.
        Equivalent to np.random.permutation()

        Parameters:
        -----------
        n : int
            Length of permutation

        Returns:
        --------
        np.ndarray
            Array of permuted numbers from 1 to n
        """
        return np.array(self.eng.randperm(float(n)))

    def __del__(self):
        """Cleanup: close MATLAB engine"""
        if hasattr(self, 'eng'):
            self.eng.quit()
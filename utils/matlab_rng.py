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
            self.seed = seed
            self.eng.eval(f'rng({seed})', nargout=0)

    def rand(self, *args):
        """
        Generate uniformly distributed random numbers using MATLAB's rand.

        Parameters:
        -----------
        *args : int or tuple
            Shape of output array. Can be:
            - No args: returns single value
            - Single number n: returns square matrix (n,n)
            - Two numbers m,n: returns matrix (m,n)
            - Tuple (m,n): returns matrix (m,n)
        """
        if len(args) > 0:
            self._check_dimensions(*args)
        if len(args) == 0:
            return float(self.eng.rand())
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
                return np.array(self.eng.rand(float(m), float(n)))
            # Single number n -> return (n,n) square matrix
            n = args[0]
            return np.array(self.eng.rand(float(n), float(n)))
        else:
            m, n = args
            return np.array(self.eng.rand(float(m), float(n)))

    def randn(self, *args):
        """
        Generate standard normal random numbers using MATLAB's randn.
        Equivalent to np.random.randn()

        Parameters:
        -----------
        *args : tuple of int
            Shape of output array. Can be:
            - No args: returns single value
            - Single number n: returns square matrix (n,n)
            - Two numbers m,n: returns matrix (m,n)
            - Tuple (m,n): returns matrix (m,n)
        """
        if len(args) > 0:
            self._check_dimensions(*args)
        if len(args) == 0:
            return float(self.eng.randn())
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                # If single number n is provided, return square matrix (n,n)
                # to match MATLAB behavior
                return np.array(self.eng.randn(float(args[0]), float(args[0])))
        else:
            m, n = args
        return np.array(self.eng.randn(float(m), float(n)))

    def normal(self, mu, sigma, *args):
        """
        Generate random numbers from normal distribution using MATLAB's normrnd.

        Parameters:
        -----------
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution
        *args : int or tuple, optional
            Shape of output array. Can be:
            - No args: returns single value
            - Single number n: returns square matrix (n,n)
            - Two numbers m,n: returns matrix (m,n)
            - Tuple (m,n): returns matrix (m,n)
        """
        # Check if sigma is positive
        if not isinstance(sigma, (int, float, np.integer, np.floating)):
            raise TypeError("Standard deviation must be a number")
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")
        
        if len(args) == 0:
            return float(self.eng.normrnd(float(mu), float(sigma)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                # Single number n, return square matrix (n,n)
                return np.array(self.eng.normrnd(float(mu), float(sigma), float(args[0]), float(args[0])))
        else:
            m, n = args
        return np.array(self.eng.normrnd(float(mu), float(sigma), float(m), float(n)))

    def uniform(self, low, high, *args):
        """
        Generate random numbers from uniform distribution using MATLAB's unifrnd.

        Parameters:
        -----------
        low : float
            Lower boundary of output interval
        high : float
            Upper boundary of output interval
        *args : int or tuple, optional
            Shape of output array. Can be:
            - No args: returns single value
            - Single number n: returns square matrix (n,n)
            - Two numbers m,n: returns matrix (m,n)
            - Tuple (m,n): returns matrix (m,n)
        """
        # Check if boundaries are valid numbers
        if not isinstance(low, (int, float, np.integer, np.floating)) or \
        not isinstance(high, (int, float, np.integer, np.floating)):
            raise TypeError("Boundaries must be numbers")
        
        # Check if high > low
        if high <= low:
            raise ValueError("Upper bound must be greater than lower bound")
    
        if len(args) == 0:
            return float(self.eng.unifrnd(float(low), float(high)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                # Single number n, return square matrix (n,n)
                return np.array(self.eng.unifrnd(float(low), float(high), float(args[0]), float(args[0])))
        else:
            m, n = args
        return np.array(self.eng.unifrnd(float(low), float(high), float(m), float(n)))

    def randi(self, imax, *args):
        """
        Generate random integers using MATLAB's randi.

        Parameters:
        -----------
        imax : int
            Upper bound of random integers (inclusive)
        *args : int or tuple, optional
            Shape of output array. Can be:
            - No args: returns single value
            - Single number n: returns square matrix (n,n)
            - Two numbers m,n: returns matrix (m,n)
            - Tuple (m,n): returns matrix (m,n)
        """
        if len(args) > 0:
            self._check_dimensions(*args)
        if len(args) == 0:
            return int(self.eng.randi(float(imax)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                # Single number n, return square matrix (n,n)
                return np.array(self.eng.randi(float(imax), float(args[0]), float(args[0])))
        else:
            m, n = args
        return np.array(self.eng.randi(float(imax), float(m), float(n)))

    def poisson(self, lam, *args):
        """
        Generate random numbers from Poisson distribution using MATLAB's poissrnd.

        Parameters:
        -----------
        lam : float
            The expected number of events occurring (lambda parameter)
        *args : int or tuple, optional
            Shape of output array. Can be:
            - No args: returns single value
            - Single number n: returns square matrix (n,n)
            - Two numbers m,n: returns matrix (m,n)
            - Tuple (m,n): returns matrix (m,n)
        """
        if len(args) > 0:
            self._check_dimensions(*args)        
        if len(args) == 0:
            return int(self.eng.poissrnd(float(lam)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                # Single number n, return square matrix (n,n)
                return np.array(self.eng.poissrnd(float(lam), float(args[0]), float(args[0])))
        else:
            m, n = args
        return np.array(self.eng.poissrnd(float(lam), float(m), float(n)))

    def randperm(self, n):
        """
        Generate random permutation using MATLAB's randperm.
        Note: randperm only takes one argument in MATLAB.

        Parameters:
        -----------
        n : int
            Length of permutation

        Returns:
        --------
        np.ndarray
            Array of permuted numbers from 1 to n with shape (1,n) to match MATLAB
        """
        # Check if n is positive integer
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")
        
        # Call MATLAB's randperm and keep the shape (1,n) to match MATLAB
        return np.array(self.eng.randperm(float(n)), dtype=float)[0]  

    def multivariate_normal(self, mean, cov, size=None):
        """
        Generate random samples from multivariate normal distribution using MATLAB's mvnrnd.

        Parameters:
        -----------
        mean : array_like
            Mean vector of the distribution
        cov : array_like
            Covariance matrix of the distribution
        size : int or tuple, optional
            Number of samples to generate. If None, returns one sample.

        Returns:
        --------
        out : ndarray
            Drawn samples from the multivariate normal distribution.
            If size is None: shape is (len(mean),)
            If size is int n: shape is (n, len(mean))

        Raises:
        -------
        ValueError:
            - If mean is not 1D array
            - If cov is not 2D array
            - If cov is not square matrix
            - If dimensions of mean and cov are incompatible
            - If cov matrix is not positive semidefinite
        TypeError:
            - If inputs are not numeric arrays
        """
        # Input validation
        mean = np.asarray(mean)
        cov = np.asarray(cov)

        if not isinstance(mean, (np.ndarray, list, tuple)):
            raise TypeError("mean must be array_like")
        if not isinstance(cov, (np.ndarray, list, tuple)):
            raise TypeError("cov must be array_like")

        # Check mean dimensions
        if mean.ndim != 1:
            raise ValueError("mean must be 1-dimensional")

        # Check covariance dimensions
        if cov.ndim != 2:
            raise ValueError("cov must be 2-dimensional")
        
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be square matrix")

        # Check compatibility of dimensions
        if mean.shape[0] != cov.shape[0]:
            raise ValueError(f"Incompatible dimensions: mean has length {mean.shape[0]}, "
                        f"cov has shape {cov.shape}")

        # Check if size is valid
        if size is not None:
            if not isinstance(size, (int, np.integer, tuple, list)):
                raise TypeError("size must be integer, tuple, or list")
            if isinstance(size, (int, np.integer)) and size <= 0:
                raise ValueError("size must be positive")
            if isinstance(size, (tuple, list)) and any(s <= 0 for s in size):
                raise ValueError("all elements in size must be positive")

        try:
            # Convert inputs to MATLAB format
            mean_matlab = matlab.double(mean.tolist())
            cov_matlab = matlab.double(cov.tolist())
            
            if size is None:
                # Generate one sample
                result = np.array(self.eng.mvnrnd(mean_matlab, cov_matlab, 1.0))
                return result.flatten()
            else:
                # Generate multiple samples
                if isinstance(size, (int, np.integer)):
                    n_samples = size
                else:
                    n_samples = np.prod(size)
                
                result = np.array(self.eng.mvnrnd(mean_matlab, cov_matlab, float(n_samples)))
                
                if isinstance(size, (tuple, list)):
                    # Reshape if size is a tuple/list
                    result = result.reshape(size + (mean.shape[0],))
                    
                return result
                
        except Exception as e:
            # Handle MATLAB errors (e.g., non-positive definite covariance matrix)
            raise ValueError(f"MATLAB mvnrnd error: {str(e)}")
    
    def __del__(self):
        """Cleanup: close MATLAB engine"""
        if hasattr(self, 'eng'):
            self.eng.quit()

    def _check_dimensions(self, *args):
        """Helper function to validate dimensions"""
        for arg in args:
            if not isinstance(arg, (int, float, np.integer, np.floating)):
                raise ValueError("Dimensions must be numeric")
            if isinstance(arg, (float, np.floating)) and not arg.is_integer():
                raise ValueError("Dimensions must be integers")
            if arg < 0:
                raise ValueError("Dimensions must be non-negative")
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
            self._set_seed(self.seed)

    def _set_seed(self, seed):
        """Reset random seed to initial value"""
        self.eng.eval(f'rng({seed})', nargout=0)

    def rand(self, *args, seed=None):
        """
        Generate uniformly distributed random numbers using MATLAB's rand.

        Parameters:
        -----------
        *args : int or tuple
            Shape of output array
        seed : int, optional
            If provided, use this seed for this specific generation
        """
        if seed is not None:
            self._set_seed(seed)

        if len(args) > 0:
            self._check_dimensions(*args)

        if len(args) == 0:
            return float(self.eng.rand())
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
                return np.array(self.eng.rand(float(m), float(n)))
            else:
                n = args[0]
                return np.array(self.eng.rand(float(n), float(n)))
        else:
            m, n = args
            return np.array(self.eng.rand(float(m), float(n)))

    def randn(self, *args, seed=None):
        """
        Generate standard normal random numbers using MATLAB's randn.

        Parameters:
        -----------
        *args : tuple of int
            Shape of output array
        seed : int, optional
            If provided, use this seed for this specific generation
        """
        if seed is not None:
            self._set_seed(seed)

        if len(args) > 0:
            self._check_dimensions(*args)

        if len(args) == 0:
            return float(self.eng.randn())
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
                return np.array(self.eng.randn(float(m), float(n)))
            else:
                return np.array(self.eng.randn(float(args[0]), float(args[0])))
        else:
            m, n = args
            return np.array(self.eng.randn(float(m), float(n)))

    def normal(self, mu, sigma, *args, seed=None):
        """
        Generate random numbers from normal distribution using MATLAB's normrnd.

        Parameters:
        -----------
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution
        *args : int or tuple, optional
            Shape of output array
        seed : int, optional
            If provided, use this seed for this specific generation
        """
        if not isinstance(sigma, (int, float, np.integer, np.floating)):
            raise TypeError("Standard deviation must be a number")
        
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")

        if seed is not None:
            self._set_seed(seed)

        if len(args) == 0:
            return float(self.eng.normrnd(float(mu), float(sigma)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                return np.array(self.eng.normrnd(float(mu), float(sigma), 
                                                    float(args[0]), float(args[0])))
        else:
            m, n = args
            return np.array(self.eng.normrnd(float(mu), float(sigma), 
                                                float(m), float(n)))

    def uniform(self, low, high, *args, seed=None):
        """
        Generate random numbers from uniform distribution using MATLAB's unifrnd.

        Parameters:
        -----------
        low : float
            Lower boundary of output interval
        high : float
            Upper boundary of output interval
        *args : int or tuple, optional
            Shape of output array
        seed : int, optional
            If provided, use this seed for this specific generation
        """
        if not isinstance(low, (int, float, np.integer, np.floating)) or \
           not isinstance(high, (int, float, np.integer, np.floating)):
            raise TypeError("Boundaries must be numbers")
        
        if high <= low:
            raise ValueError("Upper bound must be greater than lower bound")

        if seed is not None:
            self._set_seed(seed)

        if len(args) == 0:
            return float(self.eng.unifrnd(float(low), float(high)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                return np.array(self.eng.unifrnd(float(low), float(high), 
                                                    float(args[0]), float(args[0])))
        else:
            m, n = args
            return np.array(self.eng.unifrnd(float(low), float(high), 
                                                float(m), float(n)))

    def randi(self, imax, *args, seed=None):
        """
        Generate random integers using MATLAB's randi.

        Parameters:
        -----------
        imax : int
            Upper bound of random integers (inclusive)
        *args : int or tuple, optional
            Shape of output array
        seed : int, optional
            If provided, use this seed for this specific generation
        """
        if seed is not None:
            self._set_seed(seed)

        if len(args) > 0:
            self._check_dimensions(*args)

        if len(args) == 0:
            return int(self.eng.randi(float(imax)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                return np.array(self.eng.randi(float(imax), 
                                                float(args[0]), float(args[0])))
        else:
            m, n = args
            return np.array(self.eng.randi(float(imax), float(m), float(n)))

    def poisson(self, lam, *args, seed=None):
        """
        Generate random numbers from Poisson distribution using MATLAB's poissrnd.

        Parameters:
        -----------
        lam : float
            The expected number of events occurring
        *args : int or tuple, optional
            Shape of output array
        seed : int, optional
            If provided, use this seed for this specific generation
        """
        if seed is not None:
            self._set_seed(seed)

        if len(args) > 0:
            self._check_dimensions(*args)

        if len(args) == 0:
            return int(self.eng.poissrnd(float(lam)))
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                m, n = args[0]
            else:
                return np.array(self.eng.poissrnd(float(lam), 
                                                    float(args[0]), float(args[0])))
        else:
            m, n = args
            return np.array(self.eng.poissrnd(float(lam), float(m), float(n)))

    def randperm(self, n, seed=None):
        """
        Generate random permutation using MATLAB's randperm.

        Parameters:
        -----------
        n : int
            Length of permutation
        seed : int, optional
            If provided, use this seed for this specific generation
        """
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")

        if seed is not None:
            self._set_seed(seed)

        return np.array(self.eng.randperm(float(n)), dtype=float)[0]
    
    def multivariate_normal(self, mean, cov, size=None, seed=None):
        """
        Generate random samples from multivariate normal distribution using MATLAB's mvnrnd.
        Works for both univariate (1D) and multivariate cases.

        Parameters:
        -----------
        mean : array_like or scalar
            Mean of the distribution. For univariate case, it can be a scalar.
            For multivariate case, it should be a 1-D array of shape (N,) where N is the dimension.
        cov : array_like or scalar
            Covariance matrix of the distribution.
            For univariate case, it can be a scalar (variance).
            For multivariate case, it should be a 2-D array of shape (N, N).
        size : int or tuple, optional
            Number of samples to generate. If None, returns one sample.
            If int, returns array of shape (size, N).
        seed : int, optional
            If provided, this seed will be used for this specific random generation.
            The generator will return to its previous state after generation.

        Returns:
        --------
        out : ndarray
            Drawn samples from the multivariate normal distribution.
            If size is None: returns array of shape (N,)
            If size is int: returns array of shape (size, N)

        Raises:
        -------
        TypeError
            If inputs are not of the correct type
        ValueError
            If covariance matrix is not symmetric or dimensions don't match
        """
        # Set seed if provided
        if seed is not None:
            self._set_seed(seed)

        # Validate size parameter
        if size is not None:
            if not isinstance(size, (int, np.integer)):
                raise TypeError("size must be an integer")
            if size <= 0:
                raise ValueError("size must be positive")

        try:
            # Handle mean input
            if isinstance(mean, (list, tuple, np.ndarray)):
                # Convert to numpy array for validation
                mean = np.asarray(mean, dtype=float)
                if mean.ndim > 1:
                    raise ValueError("mean must be a 1-D array or scalar")
                mean_matlab = matlab.double([list(map(float, mean))])  # Convert to 1xN row vector
            else:
                # Scalar case
                if not isinstance(mean, (int, float, np.number)):
                    raise TypeError("mean must be a number or array-like")
                mean_matlab = matlab.double([[float(mean)]])
            
            # Handle covariance input
            if isinstance(cov, (list, tuple, np.ndarray)):
                cov = np.asarray(cov, dtype=float)
                if cov.ndim == 2:
                    # Check if covariance matrix is symmetric
                    if cov.shape[0] != cov.shape[1]:
                        raise ValueError("covariance matrix must be square")
                    if not np.allclose(cov, cov.T):
                        raise ValueError("covariance matrix must be symmetric")
                    # Check if matrix dimensions match mean dimensions
                    if isinstance(mean, np.ndarray) and cov.shape[0] != mean.size:
                        raise ValueError("dimensions of mean and covariance matrix do not match")
                    # Check if covariance matrix is positive semidefinite
                    if not np.all(np.linalg.eigvals(cov) >= 0):
                        raise ValueError("covariance matrix must be positive semidefinite")
                    cov_matlab = matlab.double([list(map(float, row)) for row in cov])
                else:
                    raise ValueError("covariance must be a 2-D array or scalar")
            else:
                # Scalar case
                if not isinstance(cov, (int, float, np.number)):
                    raise TypeError("covariance must be a number or array-like")
                if float(cov) < 0:
                    raise ValueError("variance must be non-negative")
                cov_matlab = matlab.double([[float(cov)]])
            
            # Generate samples using MATLAB's mvnrnd
            if size is None:
                size = 1
            result = np.array(self.eng.mvnrnd(mean_matlab, cov_matlab, float(size)))
            
            # Handle output shape
            if size == 1:
                return result.flatten()  # Return 1-D array for single sample
            return result  # Return 2-D array for multiple samples

        except Exception as e:
            if "MATLAB" in str(e):
                raise RuntimeError(f"MATLAB mvnrnd error: {str(e)}")
            raise  

    def randperm(self, n, seed=None):
        """
        Generate random permutation using MATLAB's randperm.
        Note: randperm only takes one argument in MATLAB.

        Parameters:
        -----------
        n : int
            Length of permutation
        seed : int, optional
            If provided, this seed will be used for this specific random generation
            The generator will be restored to its previous state after generation

        Returns:
        --------
        np.ndarray
            Array of permuted numbers from 1 to n with shape (1,n) to match MATLAB
        """
        # Check if n is positive integer
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")
        
        # If seed is provided, save current state and set new seed
        if seed is not None:
            self._set_seed(seed)

        # Call MATLAB's randperm and keep the shape (1,n) to match MATLAB
        return np.array(self.eng.randperm(float(n)), dtype=float)[0]

    def poisson(self, lam, *args, seed=None):
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
        seed : int, optional
            If provided, this seed will be used for this specific random generation
            The generator will be restored to its previous state after generation
        """
        if len(args) > 0:
            self._check_dimensions(*args)

        if seed is not None:
            self._set_seed(seed)

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
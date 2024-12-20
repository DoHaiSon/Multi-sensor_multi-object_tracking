import numpy as np

def one_line_array(array):
    """
    Format a numpy array as a string without new lines.
    """
    return np.array2string(array, separator=', ', formatter={'all':lambda x: str(x)}).replace('\n', '')
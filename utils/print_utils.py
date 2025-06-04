import builtins
import functools

def setup_print(verbose=True):
    """
    Set up global print function based on verbose flag.
    
    Args:
        verbose (bool, optional): Whether to enable printing. Defaults to True.
            If False, all print statements will be suppressed globally
    
    Returns:
        function: Original print function in case restoration is needed
    """
    # Store original print
    original_print = builtins.print
    
    # Create new print function
    @functools.wraps(original_print)
    def verbose_print(*args_, **kwargs):
        if verbose:
            original_print(*args_, **kwargs)
    
    # Replace built-in print with new function
    builtins.print = verbose_print
    
    return original_print  # Return original print in case we need to restore it
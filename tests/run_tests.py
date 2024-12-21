import pytest
import sys
import os

if __name__ == "__main__":
    # Get the directory containing run_tests.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add arguments for different test configurations
    args = [
        "-v",              # verbose output
        "--tb=short",      # shorter traceback format
        "-s",              # show print statements
        "--color=yes",     # colored output
        current_dir,       # run tests from current directory
    ]
    
    # Add specific test file if provided
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        # If specific test file provided, use it instead of current_dir
        args[-1] = os.path.join(current_dir, test_file)
    
    # Run pytest with the arguments
    sys.exit(pytest.main(args))
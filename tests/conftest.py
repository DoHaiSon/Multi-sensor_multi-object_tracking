import pytest
from config import get_args
from utils.logger import Logger
from utils.matlab_rng import Matlab_RNG

@pytest.fixture(scope="session")
def default_args():
    args = get_args([])
    args.model = 'Basic'
    args.enable_logging = False
    return args

@pytest.fixture(scope="session")
def logger():
    return Logger(enable_logging=False)

@pytest.fixture(scope="session")
def rng():
    return Matlab_RNG(seed=2808)
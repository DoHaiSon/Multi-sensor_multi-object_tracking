"""
Scenario 1: Multiple targets with different birth/death times
"""

import numpy as np

def get_scenario():
    """
    Returns:
        dict: Dictionary containing scenario parameters
    """
    # Constants
    nbirths = 18
    wturn = 2 * np.pi / 180

    # Initial states matrix (5 x nbirths)
    xstart = np.array([
        [1000 + 3.8676, -10, 1500 - 11.7457, -10, wturn / 8],
        [-250 - 5.8857, 20, 1000 + 11.4102, 3, -wturn / 3],
        [-1500 - 7.3806, 11, 250 + 6.7993, 10, -wturn / 2],
        [-1500, 43, 1500, 0, 0],
        [250 - 3.8676, 11, 750 - 11.0747, 5, wturn / 4],
        [1000 + 3.8676, -8, 1500 - 11.7457, -10, wturn / 6],
        [-250, 15, 1000 + 11.4102, 3, -wturn / 3],
        [-250 + 7.3806, -12, 1000 - 6.7993, -12, wturn / 2],
        [1000, 0, 1500, -10, wturn / 4],
        [250, -50, 750, 0, -wturn / 4],
        [1000, -20, 1500, -30, -wturn / 6],
        [1300, -40, 700, 20, wturn / 8],
        [1000, -50, 1500, 0, -wturn / 4],
        [250, -40, 750, 25, wturn / 4],
        [1100, -40, 1300, -10, -wturn / 3],
        [500, -60, 750, 25, wturn / 4],
        [1200, -50, 1500, 0, -wturn / 4],
        [250, -40, 750, 30, wturn / 4]
    ]).T

    # Birth and death times
    tbirth = np.array([1, 1, 10, 10, 20, 30, 30, 40, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80])
    tdeath = np.array([101, 101, 101, 66, 80, 70, 80, 101, 101, 80, 90, 101, 101, 101, 90, 101, 90, 101])

    return {
        'nbirths': nbirths,
        'wturn': wturn,
        'xstart': xstart,
        'tbirth': tbirth,
        'tdeath': tdeath
    }
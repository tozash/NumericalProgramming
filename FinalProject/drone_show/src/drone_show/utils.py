"""
Utility functions for the Drone Show Simulation.
"""
import random
import numpy as np
from . import config

def set_deterministic_behavior(seed: int = config.RANDOM_SEED):
    """
    Sets the random seed for Python's random module and NumPy's random generator
    to ensure deterministic behavior.
    
    Args:
        seed (int): The seed value to use. Defaults to config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)

import sys
from pathlib import Path

# Add src to sys.path for testing purposes if not installed
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from drone_show import config, utils

def test_imports():
    """Check that we can import the package and access config variables."""
    assert config.RANDOM_SEED == 42
    assert config.DEFAULT_N_DRONES > 0

def test_deterministic_behavior():
    """Check that setting the seed produces reproducible random numbers."""
    utils.set_deterministic_behavior(123)
    import numpy as np
    val1 = np.random.rand()
    
    utils.set_deterministic_behavior(123)
    val2 = np.random.rand()
    
    assert val1 == val2

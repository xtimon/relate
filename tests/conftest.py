"""
Shared fixtures and configuration for RELATE tests.

All tests use deterministic randomness by default.
"""

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _global_seed():
    """Reset random seeds before every test for reproducibility."""
    np.random.seed(42)
    random.seed(42)
    yield

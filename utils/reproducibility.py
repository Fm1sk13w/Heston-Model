"""Fixed random seeds so notebook figures match the report."""

from __future__ import annotations

PROJECT_SEED = 42


def set_project_seed(seed: int = PROJECT_SEED) -> None:
    """Seed NumPy (and Python ``random``) for reproducible notebook runs."""
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)

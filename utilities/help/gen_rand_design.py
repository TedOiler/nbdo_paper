import numpy as np


def gen_rand_design(N: int, P: int, low: int = -1, high: int = 1) -> np.ndarray:
    return np.random.uniform(low=low, high=high, size=(N, P))

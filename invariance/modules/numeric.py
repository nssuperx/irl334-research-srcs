import numpy as np

def zscore(x: np.ndarray) -> np.ndarray:
    """
    平均0 分散1
    """
    return (x-x.mean()) / x.std()

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """
    最小値0 最大値1
    """
    return (x - x.min()) / (x.max() - x.min())

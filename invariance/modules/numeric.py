import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    """
    平均0 分散1
    """
    return (x - x.mean()) / x.std()


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """
    最小値0 最大値1
    """
    return (x - x.min()) / (x.max() - x.min())


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """相関係数を計算（非推奨）
    通常はnp.corrcoef()を使用
    引数は要素数が等しいnumpy配列

    Args:
        x (np.ndarray): numpy配列
        y (np.ndarray): numpy配列

    Returns:
        float: 相関係数 r
    """
    # return (np.mean(x * y) - x.mean() * y.mean()) / (x.std() * y.std())
    return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())
    # return np.corrcoef(x.flatten(), y.flatten())[0][1]
    # return np.mean(x * y)

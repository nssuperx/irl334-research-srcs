from dataclasses import dataclass
import numpy as np
from typing import Tuple
from .vector2 import Vector2


@dataclass
class FciResult:
    pos: Vector2
    fci: float
    right_activity: float
    left_activity: float

    def get_all_result(self) -> Tuple[Vector2, float, float, float]:
        pass


@dataclass
class FciResultBlock:
    """
    結果をまとめて返すためだけのクラス
    2次元で持っておく意義がなくなったら使わない
    NOTE: DataFrameで代替可能
    """
    crfy: np.ndarray
    crfx: np.ndarray
    fci: np.ndarray
    rr: np.ndarray      # right RF activity
    lr: np.ndarray      # left RF activity
    raposy: np.ndarray  # right active pos y
    raposx: np.ndarray  # right active pos x
    laposy: np.ndarray  # left active pos y
    laposx: np.ndarray  # left active pos x

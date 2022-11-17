from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
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
    crfy: NDArray[np.uint32]
    crfx: NDArray[np.uint32]
    fci: NDArray[np.floating]
    rr: NDArray[np.floating]      # right RF activity
    lr: NDArray[np.floating]      # left RF activity
    raposy: NDArray[np.uint32]  # right active pos y
    raposx: NDArray[np.uint32]  # right active pos x
    laposy: NDArray[np.uint32]  # left active pos y
    laposx: NDArray[np.uint32]  # left active pos x
    overlapPixels: NDArray[np.uint32]

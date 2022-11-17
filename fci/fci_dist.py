import sys
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

from modules.io import FciDataManager


default_dataset = 1
args = sys.argv


def main():
    if (len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset
    dataMgr: FciDataManager = FciDataManager(dataset_number)
    df: pd.DataFrame = pd.read_pickle(f"{dataMgr.out_dirpath}/results_cluster.pkl")

    fci: NDArray[np.float32] = df["raw fci"].to_numpy()

    hist_bins = 100
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # NOTE: nはヒストグラムのビンの値．
    n, _, patches = ax.hist(fci, bins=hist_bins)

    # ゼロでない（テンプレートが重なっている部分の）分布の最大値
    nonZeroMax = np.sort(n)[-2]

    # 完全に一致してる部分のfciの値を見つけて，色をつける
    max_rl_activity: float = df["R L"].to_numpy().argmax()
    watch_value: float = fci[max_rl_activity]   # ここの値を注目したい値に変更してもよい
    watch_point: int = int(linear_map(watch_value, fci.min(), fci.max(), 0, hist_bins - 1))
    patches[watch_point].set_facecolor('orange')

    print(f"watch value: {watch_value}")
    print(f"fci size: {fci.size}")
    print(f"bin index: {watch_point}")

    ax.set_ylim(0, nonZeroMax * 1.5)  # ゼロ（テンプレートが重なってない）が大きすぎて見にくいため
    ax.set_xlabel("raw fci")
    ax.set_title("fci histogram")
    plt.show()


# 参考: https://www.arduino.cc/reference/en/language/functions/math/map/
def linear_map(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if __name__ == "__main__":
    main()

import sys
import numpy as np
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

    fci: np.ndarray = df["raw fci"].to_numpy()

    # 完全に一致してるとき
    max_rl_activity: float = df["R L"].to_numpy().argmax()
    print(f"watch value: {fci[max_rl_activity]}")
    print(f"fci size: {fci.size}")

    hist_bins = 100
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    _, _, patches = ax.hist(fci, bins=hist_bins)

    watch_value: float = fci[max_rl_activity]
    watch_point: int = int(watch_value * hist_bins / fci.max())
    patches[watch_point].set_facecolor('orange')
    ax.set_xlabel("raw fci")
    ax.set_title("fci histogram")
    plt.show()


if __name__ == "__main__":
    main()

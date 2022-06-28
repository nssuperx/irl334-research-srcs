import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.io import FciDataManager


def main():
    dataMgr: FciDataManager = FciDataManager(1)
    df: pd.DataFrame = pd.read_pickle(f"{dataMgr.get_out_dirpath()}/results.pkl")

    fci: np.ndarray = df["fci"].to_numpy()

    hist_bins = 100
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    _, _, patches = ax.hist(fci, bins=hist_bins)

    watch_value: float = 0.884543650386626
    watch_point: int = int(watch_value * hist_bins / fci.max())
    patches[watch_point].set_facecolor('orange')
    ax.set_xlabel("raw fci")
    ax.set_title("fci histogram")
    plt.show()


if __name__ == "__main__":
    main()

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

    watch_point: float = 0.884543650386626
    watch_point_idx: int = int(watch_point * hist_bins / fci.max())
    patches[watch_point_idx].set_facecolor('orange')
    plt.show()


if __name__ == "__main__":
    main()

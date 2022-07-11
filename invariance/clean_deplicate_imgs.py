import os
import shutil
import pandas as pd

from modules.io import FciDataManager


def main():
    dataMgr: FciDataManager = FciDataManager(1)
    df = pd.read_csv(f"{dataMgr.get_out_dirpath()}{os.sep}crf_cluster.csv", index_col=0)
    filenames = list(df.index)
    for name in filenames:
        shutil.copy2(f"{dataMgr.get_out_dirpath()}{os.sep}crf{os.sep}{name}.png",
                     f"{dataMgr.get_out_dirpath()}{os.sep}crf-clean{os.sep}{name}.png")


if __name__ == "__main__":
    main()

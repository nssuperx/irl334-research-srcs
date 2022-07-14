import sys
import os
import shutil
import pandas as pd
from tqdm import tqdm

from modules.io import FciDataManager

default_dataset = 1
args = sys.argv


def main():
    if(len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset
    dataMgr: FciDataManager = FciDataManager(dataset_number)
    df = pd.read_csv(f"{dataMgr.get_out_dirpath()}{os.sep}crf_cluster.csv", index_col=0)
    filenames = list(df.index)
    for name in tqdm(filenames, desc="files"):
        shutil.copy2(f"{dataMgr.get_out_dirpath()}{os.sep}crf{os.sep}{name}.png",
                     f"{dataMgr.get_out_dirpath()}{os.sep}crf-clean{os.sep}{name}.png")


if __name__ == "__main__":
    main()

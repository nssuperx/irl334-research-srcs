import sys
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.io import FciDataManager

default_dataset = 1
args = sys.argv


def main():
    # 重複部分をなくしたcsvファイルを作る
    if (len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset
    dataMgr: FciDataManager = FciDataManager(dataset_number)
    df: pd.DataFrame = pd.read_pickle(f"{dataMgr.out_dirpath}/results.pkl")

    # NOTE: 表では，right RFのほうが左側にあるので，left側のindexになる．
    left_idx = df.columns.get_loc("right RF most active y")
    right_idx = df.columns.get_loc("left RF most active x")

    pos_dict: Dict[tuple, List[tuple]] = {}

    for i, row in tqdm(df.iterrows(), desc="rows", total=len(df)):
        pos_key = tuple(np.array(row[left_idx: right_idx + 1], dtype=np.uint32))
        if pos_key not in pos_dict.keys():
            pos_dict[pos_key] = []
        pos_dict[pos_key].append((row["y"], row["x"]))

    cluster_df: pd.DataFrame = pd.DataFrame(columns=df.columns)
    for poss in tqdm(pos_dict.values(), desc="classes"):
        # NOTE: np.median()を使いたいが，要素がほしいので，ソートして中央の値を取る
        yxarray = np.sort(np.array(poss, dtype=np.int32), axis=0)
        yx = yxarray[yxarray.shape[0] // 2]
        cluster_df = pd.concat([cluster_df, df[(df["y"] == yx[0]) & (df["x"] == yx[1])]])

    cluster_df.to_csv(f"{dataMgr.out_dirpath}/crf_cluster.csv")
    cluster_df.to_pickle(f"{dataMgr.out_dirpath}/results_cluster.pkl")


if __name__ == "__main__":
    main()

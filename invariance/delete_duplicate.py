from typing import Dict, List
import numpy as np
import pandas as pd

from modules.io import FciDataManager


def main():
    dataMgr: FciDataManager = FciDataManager(1)
    df: pd.DataFrame = pd.read_pickle(f"{dataMgr.get_out_dirpath()}/results.pkl")

    left_idx = df.columns.get_loc("right RF most active y")
    right_idx = df.columns.get_loc("left RF most active x")

    pos_dict: Dict[tuple, List[tuple]] = {}

    for i, row in df.iterrows():
        pos_key = tuple(np.array(row[left_idx: right_idx + 1], dtype=np.uint32))
        if pos_key not in pos_dict.keys():
            pos_dict[pos_key] = []
        pos_dict[pos_key].append((row["y"], row["x"]))

    cluster_df: pd.DataFrame = pd.DataFrame(columns=df.columns)
    for poss in pos_dict.values():
        # NOTE: np.median()を使いたいが，要素がほしいので，ソートして中央の値を取る
        yxarray = np.sort(np.array(poss, dtype=np.int32), axis=0)
        yx = yxarray[yxarray.shape[0] // 2]
        tmp = df[(df["y"] == yx[0]) & (df["x"] == yx[1])]
        cluster_df = pd.concat([cluster_df, tmp])

    cluster_df.to_csv(f"{dataMgr.get_out_dirpath()}/crf_cluster.csv")


if __name__ == "__main__":
    main()

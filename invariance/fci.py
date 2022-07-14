import sys
import json
import numpy as np
import pandas as pd

from modules.numeric import min_max_normalize
from modules.function import scan_combinedRF
from modules.io import save_image, FciDataManager
from modules.core import TemplateImage
from modules.results import FciResultBlock


default_dataset = 3
args = sys.argv


def main():
    if(len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset
    dataMgr: FciDataManager = FciDataManager(dataset_number)
    dataMgr.load_image()
    originalImgArray = dataMgr.originalImg
    rightTemplate = TemplateImage(dataMgr.rightEyeImg)
    leftTemplate = TemplateImage(dataMgr.leftEyeImg)
    # 入力
    dataMgr.load_scan_array()
    rightScanImgArray = dataMgr.rightScanImg
    leftScanImgArray = dataMgr.leftScanImg

    with open(f"{dataMgr.get_dirpath()}/setting.json", "r") as f:
        jsondata = json.load(f)

    setting = jsondata["default"]
    height = int(setting["height"])
    width = int(setting["width"])
    crf_width = int(setting["crf_width"])
    scanStep = int(setting["scanstep"])

    # 全部scanしてみる
    res: FciResultBlock = scan_combinedRF(height, crf_width, height, width, scanStep, originalImgArray,
                                          rightScanImgArray, rightTemplate, leftScanImgArray, leftTemplate,
                                          dataMgr, saveImage=True)
    print(f"fci mean: {res.fci.mean()}")
    print(f"fci std: {res.fci.std()}")
    print(f"max fci pos: {np.unravel_index(np.argmax(res.fci), res.fci.shape)}")

    fciindex = [f"y{y:04}x{x:04}" for y, x in zip(res.crfy.flatten(), res.crfx.flatten())]

    # この処理は適当に思いついたもの
    res.fci[res.fci < 0.0] = 0.0
    # fci = np.where((fci < 0.0, 0.0, fci))
    raw_fci = res.fci
    fci = min_max_normalize(res.fci)
    rrlr = res.rr * res.lr
    rrlrfci = rrlr * fci

    resultData = np.vstack([res.crfy.flatten(), res.crfx.flatten(),
                            raw_fci.flatten(), fci.flatten(), res.rr.flatten(), res.lr.flatten(),
                            rrlr.flatten(), rrlrfci.flatten(),
                            res.raposy.flatten(), res.raposx.flatten(), res.laposy.flatten(), res.laposx.flatten()]).T
    df = pd.DataFrame(resultData, index=fciindex,
                      columns=["y", "x", "raw fci", "fci", "cell R activity", "cell L activity", "R L", "R L fci",
                               "right RF most active y", "right RF most active x",
                               "left RF most active y", "left RF most active x"])
    df.to_csv(f"{dataMgr.get_out_dirpath()}/crf_skip{scanStep}.csv")
    df.to_pickle(f"{dataMgr.get_out_dirpath()}/results.pkl")
    save_image(f"{dataMgr.get_out_dirpath()}/fciImg.png", fci)


if __name__ == "__main__":
    main()

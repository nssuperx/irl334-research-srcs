import numpy as np
import pandas as pd

from modules.numeric import min_max_normalize
from modules.image.function import scan_combinedRF
from modules.image.io import save_image, FciDataManager
from modules.core import TemplateImage
from modules.results import FciResultBlock


def main():
    data = FciDataManager(1)
    data.load_image()
    originalImgArray = data.originalImg
    rightTemplate = TemplateImage(data.rightEyeImg)
    leftTemplate = TemplateImage(data.leftEyeImg)
    # 入力
    data.load_scan_array()
    rightScanImgArray = data.rightScanImg
    leftScanImgArray = data.leftScanImg
    # rightScanImgArray = zscore(rightScanImgArray)
    # leftScanImgArray = zscore(leftScanImgArray)

    # テスト: 一つReceptiveFieldを作る
    # height: 30, width: 30, overlap: 12
    height = 70
    width = 70
    crf_width = 110
    scanStep = 4

    # 全部scanしてみる
    res: FciResultBlock = scan_combinedRF(height, crf_width, height, width, scanStep, originalImgArray,
                                          rightScanImgArray, rightTemplate, leftScanImgArray, leftTemplate)
    print(f"fci mean: {res.fci.mean()}")
    print(f"fci std: {res.fci.std()}")
    print(f"max fci pos: {np.unravel_index(np.argmax(res.fci), res.fci.shape)}")

    fciindex = [f"y{y*scanStep:04}x{x*scanStep:04}" for y in range(res.fci.shape[0]) for x in range(res.fci.shape[1])]

    # この処理は適当に思いついたもの
    res.fci[res.fci < 0.0] = 0.0
    # fci = np.where((fci < 0.0, 0.0, fci))
    fci = min_max_normalize(res.fci)
    rrlr = res.rr * res.lr
    rrlrfci = rrlr * fci

    resultData = np.vstack([fci.flatten(), res.rr.flatten(), res.lr.flatten(), rrlr.flatten(), rrlrfci.flatten(),
                            res.raposy.flatten(), res.raposx.flatten(), res.laposy.flatten(), res.laposx.flatten()]).T
    data = pd.DataFrame(resultData, index=fciindex,
                        columns=["fci", "cell R activity", "cell L activity", "R L", "R L fci",
                                 "right RF most active y", "right RF most active x",
                                 "left RF most active y", "left RF most active x"])
    data.to_csv(f"./test_skip{scanStep}.csv")
    save_image("./fci.png", fci)


if __name__ == "__main__":
    main()

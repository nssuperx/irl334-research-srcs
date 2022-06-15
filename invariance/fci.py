import numpy as np
import pandas as pd

from modules.numeric import min_max_normalize
from modules.image.function import scan_combinedRF
from modules.image.io import save_image, FciDataManager
from modules.core import TemplateImage


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
    overlap = 30
    crf_width = 110
    noOverlap = crf_width - width

    # 全部scanしてみる
    fci, rr, lr = scan_combinedRF(height, crf_width, height, width, 1, originalImgArray,
                                  rightScanImgArray, rightTemplate, leftScanImgArray, leftTemplate)
    print(f"fci mean: {fci.mean()}")
    print(f"fci std: {fci.std()}")
    print(f"max fci pos: {np.unravel_index(np.argmax(fci), fci.shape)}")

    fciindex = [f"y{y:04}x{x:04}" for y in range(fci.shape[0]) for x in range(fci.shape[1])]

    # この処理は適当に思いついたもの
    fci[fci < 0.0] = 0.0
    # fci = np.where((fci < 0.0, 0.0, fci))
    fci = min_max_normalize(fci)
    rrlr = rr * lr
    rrlrfci = rrlr * fci

    resultData = np.vstack([fci.flatten(), rr.flatten(), lr.flatten(), rrlr.flatten(), rrlrfci.flatten()]).T
    data = pd.DataFrame(resultData, index=fciindex,
                        columns=["fci", "cell R activity", "cell L activity", "R L", "R L fci"])
    data.to_csv("./test.csv")
    save_image("./fci.png", fci)


if __name__ == "__main__":
    main()

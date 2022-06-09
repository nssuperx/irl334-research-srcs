import numpy as np

from modules.image.function import scan_combinedRF
from modules.image.io import load_image, save_image
from modules.core import TemplateImage


def main():
    imgDic = load_image(2)
    originalImgArray = imgDic["original"]
    rightTemplate = TemplateImage(imgDic["right_eye"])
    leftTemplate = TemplateImage(imgDic["left_eye"])
    # 入力
    rightScanImgArray = np.load("./rightScanImgArray.npy")
    leftScanImgArray = np.load("./leftScanImgArray.npy")
    # rightScanImgArray = zscore(rightScanImgArray)
    # leftScanImgArray = zscore(leftScanImgArray)

    # テスト: 一つReceptiveFieldを作る
    # height: 30, width: 30, overlap: 12
    height = 30
    width = 30
    overlap = 12
    crf_width = 48
    noOverlap = crf_width - width

    # 全部scanしてみる
    fci = scan_combinedRF(height, crf_width, height, width, 1, originalImgArray, rightScanImgArray, rightTemplate, leftScanImgArray, leftTemplate)
    save_image("./fci.png", fci)


if __name__ == "__main__":
    main()

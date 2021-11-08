from typing import ContextManager, Dict, Tuple
import numpy as np

from modules.core import TemplateImage, ReceptiveField, CombinedReceptiveField
from modules.image.io import load_image, save_image
from modules.image.function import scan, scan_combinedRF
from modules.numeric import zscore, min_max_normalize

"""
テストの仕方
全部反転させれば，一番相関が高かったところが低くなる
"""

def main():
    imgDic = load_image()
    # image_read_test(imgDic)
    originalImgArray = imgDic["original"]
    # rightEyeImgArray = imgDic["right_eye"]
    # leftEyeImgArray = imgDic["left_eye"]
    rightTemplate = TemplateImage(imgDic["right_eye"])
    leftTemplate = TemplateImage(imgDic["left_eye"])

    # 走査した画像配列を作成
    # rightScanImgArray = scan(originalImgArray, rightTemplate.img)
    # leftScanImgArray = scan(originalImgArray, leftTemplate.img)
    # save_image("./images/out/rightScanImg.png", rightScanImgArray)
    # save_image("./images/out/leftScanImg.png", leftScanImgArray)

    # rightScanImgArray = zscore(rightScanImgArray)
    # leftScanImgArray = zscore(leftScanImgArray)
    # np.save("./rightScanImgTmp", rightScanImgArray)
    # np.save("./leftScanImgTmp", leftScanImgArray)
    
    # 入力
    rightScanImgArray = np.load("./rightScanImgTmp.npy")
    leftScanImgArray = np.load("./leftScanImgTmp.npy")

    # 試しに一つReceptiveFieldを作る
    # test_one_cRF(rightScanImgArray, leftScanImgArray, rightTemplate, leftTemplate)

    # 全部scanしてみる
    # fciArray = scan_combinedRF(70, 110, 1, originalImgArray, rightScanImgArray, rightTemplate, leftScanImgArray, leftTemplate)
    # np.save("./fciArray", fciArray)
    fciArray = np.load("./fciArray.npy")
    # print(fciArray)

if __name__ == "__main__":
    main()

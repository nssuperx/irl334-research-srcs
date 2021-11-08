from typing import ContextManager, Dict, Tuple
import numpy as np

from modules.core import TemplateImage, ReceptiveField, CombinedReceptiveField
from modules.image.io import read_image, save_image

"""
テストの仕方
全部反転させれば，一番相関が高かったところが低くなる
"""

def main():
    imgDic = read_image()
    # image_read_test(imgDic)
    originalImgArray = imgDic["original"]
    # rightEyeImgArray = imgDic["right_eye"]
    # leftEyeImgArray = imgDic["left_eye"]
    rightTemplate = TemplateImage(imgDic["right_eye"])
    leftTemplate = TemplateImage(imgDic["left_eye"])

    # 走査した画像配列を作成
    # rightScanImgArray = scan(originalImgArray, rightEyeImgArray)
    # leftScanImgArray = scan(originalImgArray, leftEyeImgArray)

    # 入力
    rightScanImgArray = np.load("./rightScanImgTmp.npy")
    leftScanImgArray = np.load("./leftScanImgTmp.npy")

    # 試しに一つReceptiveFieldを作る
    # test_one_cRF(rightScanImgArray, leftScanImgArray, rightTemplate, leftTemplate)

    # 全部scanしてみる
    rfHeight = 70
    rfWidth = 70
    cRFHeight = 70
    cRFWidth = 110
    step = 10
    fciArray = np.zeros((originalImgArray.shape[0] // step, originalImgArray.shape[1] // step))
    for y in range(0, cRFHeight, step):
        for x in range(0, cRFWidth, step):
            rightRF = ReceptiveField((y, x), rightScanImgArray, rightTemplate)
            leftRF = ReceptiveField((y, x + (cRFWidth- cRFHeight)), leftScanImgArray, leftTemplate)
            combinedRF = CombinedReceptiveField(rightRF, leftRF)
            fciArray[y//step][x//step] = combinedRF.get_fci()
    # 適当に正規化([0,1])
    fciArray = (fciArray - fciArray.mean()) / fciArray.std()

    # np.save("./rightScanImgTmp", rightScanImgArray)
    save_image("./images/out/rightScanImg.png", rightScanImgArray)
    # np.save("./leftScanImgTmp", leftScanImgArray)
    save_image("./images/out/leftScanImg.png", leftScanImgArray)

if __name__ == "__main__":
    main()

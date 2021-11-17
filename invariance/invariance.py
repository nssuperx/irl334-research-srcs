from typing import ContextManager, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# use latex style
# from matplotlib import rc
# rc('text', usetex=True)

from modules.core import TemplateImage, ReceptiveField, CombinedReceptiveField
from modules.image.io import load_image, save_image
from modules.image.function import scan, scan_combinedRF
from modules.numeric import zscore, min_max_normalize
from modules.test import image_read_test

"""
テストの仕方
全部反転させれば，一番相関が高かったところが低くなる
"""

def main():
    """
    以下，使用例まとめ

    imgDic = load_image(1)
    # image_read_test(imgDic)
    originalImgArray = imgDic["original"]
    rightTemplate = TemplateImage(imgDic["right_eye"])
    leftTemplate = TemplateImage(imgDic["left_eye"])

    # 走査した画像配列を作成
    # rightScanImgArray = scan(originalImgArray, rightTemplate.img)
    # leftScanImgArray = scan(originalImgArray, leftTemplate.img)
    # save_image("./images/out/rightScanImg.png", rightScanImgArray)
    # save_image("./images/out/leftScanImg.png", leftScanImgArray)

    # rightScanImgArray = zscore(rightScanImgArray)
    # leftScanImgArray = zscore(leftScanImgArray)
    # np.save("./rightScanImgArray", rightScanImgArray)
    # np.save("./leftScanImgArray", leftScanImgArray)
    
    # 入力
    rightScanImgArray = np.load("./rightScanImgArray.npy")
    leftScanImgArray = np.load("./leftScanImgArray.npy")

    # 試しに一つReceptiveFieldを作る
    # test_one_cRF(rightScanImgArray, leftScanImgArray, rightTemplate, leftTemplate)

    # 全部scanしてみる
    # fciArray = scan_combinedRF(70, 110, 1, originalImgArray, rightScanImgArray, rightTemplate, leftScanImgArray, leftTemplate)
    # np.save("./fciArray", fciArray)
    # fciArray = np.load("./fciArray.npy")
    # print(fciArray)

    # ヒストグラム描画
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(rightScanImgArray.flatten(), bins=100, normed=True, alpha=0.5)
    ax.hist(leftScanImgArray.flatten(), bins=100, normed=True, alpha=0.5)
    # ax.set_xlabel(r'$r$')
    ax.set_xlabel('r')
    plt.show()
    """

    for i in range(1, 3 + 1):
        imgDic = load_image(i)
        # image_read_test(imgDic)
        originalImgArray = imgDic["original"]
        rightTemplate = TemplateImage(imgDic["right_eye"])
        leftTemplate = TemplateImage(imgDic["left_eye"])

        # 走査した画像配列を作成
        rightScanImgArray = scan(originalImgArray, rightTemplate.img)
        leftScanImgArray = scan(originalImgArray, leftTemplate.img)
        save_image("./dataset/" + str(i) + "/out/rightScanImg.png", rightScanImgArray)
        save_image("./dataset/" + str(i) + "/out/leftScanImg.png", leftScanImgArray)

        # rightScanImgArray = zscore(rightScanImgArray)
        # leftScanImgArray = zscore(leftScanImgArray)
        np.save("./dataset/" + str(i) + "/array/rightScanImg", rightScanImgArray)
        np.save("./dataset/" + str(i) + "/array/leftScanImg", leftScanImgArray)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(rightScanImgArray.flatten(), bins=100, alpha=0.5, label="right")
        ax.hist(leftScanImgArray.flatten(), bins=100, alpha=0.5, label="left")
        # ax.set_xlabel(r'$r$')
        ax.legend()
        ax.set_xlabel('r')
        # plt.show()
        plt.savefig("./dataset/" + str(i) + "/out/hist.pdf")

if __name__ == "__main__":
    main()

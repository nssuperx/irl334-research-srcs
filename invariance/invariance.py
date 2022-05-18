from typing import ContextManager, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

# use latex style
# from matplotlib import rc
# rc('text', usetex=True)

from modules.core import TemplateImage, ReceptiveField, CombinedReceptiveField
from modules.image.io import load_image, save_image
from modules.image.function import scan, scan_combinedRF
from modules.numeric import zscore, min_max_normalize
from modules.test import image_read_test

def main():
    # 以下，使用例まとめ

    imgDic = load_image(2)
    # image_read_test(imgDic)
    originalImgArray = imgDic["original"]
    rightTemplate = TemplateImage(imgDic["right_eye"])
    leftTemplate = TemplateImage(imgDic["left_eye"])

    # 走査した画像配列を作成
    # rightScanImgArray = scan(originalImgArray, rightTemplate.img)
    # leftScanImgArray = scan(originalImgArray, leftTemplate.img)
    # save_image("./images/out/rightScanImg.png", rightScanImgArray)
    # save_image("./images/out/leftScanImg.png", leftScanImgArray)

    # np.save("./rightScanImgArray", rightScanImgArray)
    # np.save("./leftScanImgArray", leftScanImgArray)
    
    # 入力
    rightScanImgArray = np.load("./rightScanImgArray.npy")
    leftScanImgArray = np.load("./leftScanImgArray.npy")
    rightScanImgArray = zscore(rightScanImgArray)
    leftScanImgArray = zscore(leftScanImgArray)

    # テスト: 一つReceptiveFieldを作る
    noOverlap = 70 - (70 * 2 - 110)
    test_leftRF = ReceptiveField((0,0), leftScanImgArray, leftTemplate)
    test_leftRF.show_img(originalImgArray)
    test_rightRF = ReceptiveField((0,noOverlap), rightScanImgArray, rightTemplate)
    test_rightRF.show_img(originalImgArray)

    # テスト: 一つCombinedRF
    test_crf = CombinedReceptiveField(test_leftRF, test_rightRF)
    im = Image.fromarray(min_max_normalize(originalImgArray[0:test_crf.height, 0:test_crf.width]) * 255).convert('L')
    draw = ImageDraw.Draw(im)
    draw.rectangle((test_leftRF.mostActivePos[1], test_leftRF.mostActivePos[0], test_leftRF.mostActivePos[1] + leftTemplate.img.shape[1], test_leftRF.mostActivePos[0] + leftTemplate.img.shape[0]))
    draw.rectangle((test_rightRF.mostActivePos[1] + noOverlap, test_rightRF.mostActivePos[0], test_rightRF.mostActivePos[1] + rightTemplate.img.shape[1] + noOverlap, test_rightRF.mostActivePos[0] + rightTemplate.img.shape[0]))
    # im.show()

    exit()

    # 全部scanしてみる
    fciArray = scan_combinedRF(70, 110, 1, originalImgArray, rightScanImgArray, rightTemplate, leftScanImgArray, leftTemplate)
    np.save("./fciArray", fciArray)
    fciArray = np.load("./fciArray.npy")
    # print(fciArray)

    # ヒストグラム描画
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(rightScanImgArray.flatten(), bins=100, normed=True, alpha=0.5)
    ax.hist(leftScanImgArray.flatten(), bins=100, normed=True, alpha=0.5)
    # ax.set_xlabel(r'$r$')
    ax.set_xlabel('r')
    plt.show()
    """

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
    """

if __name__ == "__main__":
    main()

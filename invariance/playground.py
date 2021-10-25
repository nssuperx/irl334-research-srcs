from typing import ContextManager, Dict
from PIL import Image
import numpy as np

"""
テストの仕方
全部反転させれば，一番相関が高かったところが低くなる
"""

class TemplateImage:
    img: np.ndarray
    mean: float
    variance: float
    sd: float

    def __init__(self, imgArray: np.ndarray) -> None:
        self.img = imgArray
        self.mean = self.img.mean()
        self.variance = self.img.var()
        self.sd = self.img.std()
    

def main():
    imgDic = read_image()
    # image_read_test(imgDic)
    originalImgArray = imgDic["original"]
    rightEyeImgArray = imgDic["right_eye"]
    leftEyeImgArray = imgDic["left_eye"]
    print(originalImgArray[0])

    # 右目で走査した画像配列を作成
    rightScanImgArray = np.empty((originalImgArray.shape[0] - rightEyeImgArray.shape[0], originalImgArray.shape[1] - rightEyeImgArray.shape[1]))
    leftScanImgArray = np.empty((originalImgArray.shape[0] - leftEyeImgArray.shape[0], originalImgArray.shape[1] - leftEyeImgArray.shape[1]))
    print(rightScanImgArray.shape)

    # 走査
    # TODO: numpy.corrcoef()の挙動をちゃんと調べる
    for i in range(rightScanImgArray.shape[0]):
        for j in range(rightScanImgArray.shape[1]):
            # 相関係数出す範囲をスライス
            scanTargetImgArray = originalImgArray[i:i+rightEyeImgArray.shape[0], j:j+rightEyeImgArray.shape[1]]
            cov = np.mean(np.multiply(scanTargetImgArray, rightEyeImgArray))
            # rightScanImgArray[i][j] = np.corrcoef(scanTargetImgArray.flatten(), rightEyeImgArray.flatten())[0][1]
            rightScanImgArray[i][j] = cov / (scanTargetImgArray.std() * rightEyeImgArray.std())
    
    for i in range(leftScanImgArray.shape[0]):
        for j in range(leftScanImgArray.shape[1]):
            # 相関係数出す範囲をスライス
            scanTargetImgArray = originalImgArray[i:i+leftEyeImgArray.shape[0], j:j+leftEyeImgArray.shape[1]]
            leftScanImgArray[i][j] = np.corrcoef(scanTargetImgArray.flatten(), leftEyeImgArray.flatten())[0][1]

    # 出力
    # rightScanImgArray = np.load("./rightScanImgTmp.npy")
    np.save("./rightScanImgTmp", rightScanImgArray)
    rightimg = Image.fromarray(rightScanImgArray * 255).convert("L")
    rightimg.show()
    rightimg.save("./images/out/rightScanImg.png")

    # leftScanImgArray = np.load("./leftScanImgTmp.npy")
    np.save("./leftScanImgTmp", leftScanImgArray)
    leftimg = Image.fromarray(leftScanImgArray * 255).convert("L")
    leftimg.show()
    leftimg.save("./images/out/leftScanImg.png")



def read_image(normalize: bool = True) -> dict:
    """
    画像を読み込む

    Returns
    ----------
    Dictionary
        key: 画像の種類(original, right_eye, left_eye)
        value: numpy.adarray: 読み込んだ画像
    """
    originalImagePath = "./images/in/sample.png"
    rightEyeImagePath = "./images/in/right_eye.png"
    leftEyeImagePath = "./images/in/left_eye.png"

    if normalize:
        originalImage = np.array(Image.open(originalImagePath))
        rightEyeImage = np.array(Image.open(rightEyeImagePath))
        leftEyeImage = np.array(Image.open(leftEyeImagePath))

        originalImage = (originalImage - originalImage.mean()) / originalImage.std()
        rightEyeImage = (rightEyeImage - rightEyeImage.mean()) / rightEyeImage.std()
        leftEyeImage = (leftEyeImage - leftEyeImage.mean()) / leftEyeImage.std()
    else:
        # 画像読み込みつつndarrayに変換，asarray()を使うとread-onlyなデータができる．
        originalImage = np.asarray(Image.open(originalImagePath))
        rightEyeImage = np.asarray(Image.open(rightEyeImagePath))
        leftEyeImage = np.asarray(Image.open(leftEyeImagePath))

    imgDic = {"original": originalImage, "right_eye": rightEyeImage, "left_eye": leftEyeImage}

    return imgDic


def image_read_test(imgDic: dict):
    """
    画像を読み込めたかテストする

    Parameters
    ----------
    Dictionary
        key: 画像の種類(original, right_eye, left_eye)
        value: numpy.adarray: 読み込んだ画像
    """
    print(type(imgDic))

    for key, value in imgDic.items():
        print("type:" + str(type(value)) + " " + str(key) + " shape:" + str(value.shape))

    """
    for imgArray in imgDic.values():
        img = Image.fromarray(imgArray)
        img.show()
    """

if __name__ == "__main__":
    main()

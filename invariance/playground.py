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


def scan(originalImgArray: np.ndarray, templateImgArray: np.ndarray) -> np.ndarray:
    # 走査
    # TODO: 遅すぎるのでなんとかする
    scanImgArray = np.empty((originalImgArray.shape[0] - templateImgArray.shape[0], originalImgArray.shape[1] - templateImgArray.shape[1]))
    for i in range(scanImgArray.shape[0]):
        for j in range(scanImgArray.shape[1]):
            # 相関係数出す範囲をスライス
            scanTargetImgArray = originalImgArray[i:i+templateImgArray.shape[0], j:j+templateImgArray.shape[1]]
            cov = np.mean(np.multiply(scanTargetImgArray, templateImgArray))
            # scanImgArray[i][j] = np.corrcoef(scanTargetImgArray.flatten(), templateImgArray.flatten())[0][1]
            scanImgArray[i][j] = cov / (scanTargetImgArray.std() * templateImgArray.std())

    return scanImgArray


def image_save(filepath: str, scanImgArray: np.ndarray) -> None:
    img = Image.fromarray(scanImgArray * 255).convert("L")
    # img.show()
    img.save(filepath)
    

def main():
    imgDic = read_image()
    # image_read_test(imgDic)
    originalImgArray = imgDic["original"]
    rightEyeImgArray = imgDic["right_eye"]
    leftEyeImgArray = imgDic["left_eye"]

    # 走査した画像配列を作成
    # rightScanImgArray = scan(originalImgArray, rightEyeImgArray)
    # leftScanImgArray = scan(originalImgArray, leftEyeImgArray)

    # 出力
    rightScanImgArray = np.load("./rightScanImgTmp.npy")
    # np.save("./rightScanImgTmp", rightScanImgArray)
    image_save("./images/out/rightScanImg.png", rightScanImgArray)

    leftScanImgArray = np.load("./leftScanImgTmp.npy")
    # np.save("./leftScanImgTmp", leftScanImgArray)
    image_save("./images/out/leftScanImg.png", leftScanImgArray)


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

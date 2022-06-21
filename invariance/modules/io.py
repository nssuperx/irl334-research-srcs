from PIL import Image
import numpy as np
from .numeric import min_max_normalize, zscore


class FciDataManager:
    def __init__(self, setNumber: int) -> None:
        self.setNumber = setNumber

    def load_image(self) -> None:
        imgs = load_image(self.setNumber)
        self.originalImg: np.ndarray = imgs["original"]
        self.rightEyeImg: np.ndarray = imgs["right_eye"]
        self.leftEyeImg: np.ndarray = imgs["left_eye"]

    def load_scan_array(self) -> None:
        self.rightScanImg: np.ndarray = np.load(f"./dataset/{self.setNumber}/array/rightScanImg.npy")
        self.leftScanImg: np.ndarray = np.load(f"./dataset/{self.setNumber}/array/leftScanImg.npy")

    def save_scan_image(self, rightScanImg: np.ndarray, leftScanImg: np.ndarray, dirpath: str = None) -> None:
        if dirpath is None:
            dirpath = f"./dataset/{self.setNumber}/out/"
        rightimg = Image.fromarray(min_max_normalize(rightScanImg) * 255).convert("L")
        leftimg = Image.fromarray(min_max_normalize(leftScanImg) * 255).convert("L")
        rightimg.save(f"{dirpath}rightScanImg.png")
        leftimg.save(f"{dirpath}leftScanImg.png")

    def save_scan_array(self, rightScanImg: np.ndarray, leftScanImg: np.ndarray, dirpath: str = None) -> None:
        if dirpath is None:
            dirpath = f"./dataset/{self.setNumber}/array/"
        np.save(f"{dirpath}rightScanImg", rightScanImg)
        np.save(f"{dirpath}leftScanImg", leftScanImg)


def save_image(filepath: str, scanImg: np.ndarray) -> None:
    """（非推奨）
    画像を保存する
    この関数内で正規化を行うので，元画像配列の状態は気にしなくてok

    Args:
        filepath (str): 保存するパス
        scanImg (np.ndarray): 保存したい画像配列
    """
    img = Image.fromarray(min_max_normalize(scanImg) * 255).convert("L")
    # img.show()
    img.save(filepath)


def load_image(setNumber: int, normalize: bool = True) -> dict:
    """画像を読み込む

    Args:
        setNumber (int): データセット番号
        normalize (bool, optional): 正規化するかしないか. Defaults to True.

    Returns:
        dict: 3種類の画像が格納されているDictionary
            key: 画像の種類(original, right_eye, left_eye)
            value: numpy.ndarray: 読み込んだ画像
    """

    originalImgPath = f"./dataset/{setNumber}/in/original.png"
    rightEyeImgPath = f"./dataset/{setNumber}/in/right_eye.png"
    leftEyeImgPath = f"./dataset/{setNumber}/in/left_eye.png"

    if normalize:
        originalImg = zscore(np.array(Image.open(originalImgPath)))
        rightEyeImg = zscore(np.array(Image.open(rightEyeImgPath)))
        leftEyeImg = zscore(np.array(Image.open(leftEyeImgPath)))

    else:
        originalImg = np.array(Image.open(originalImgPath))
        rightEyeImg = np.array(Image.open(rightEyeImgPath))
        leftEyeImg = np.array(Image.open(leftEyeImgPath))

    imgDic = {"original": originalImg, "right_eye": rightEyeImg, "left_eye": leftEyeImg}

    return imgDic

from PIL import Image
import numpy as np
from ..numeric import min_max_normalize

def save_image(filepath: str, scanImg: np.ndarray) -> None:
    """画像を保存する
    この関数内で正規化を行うので，元画像配列の状態は気にしなくてok

    Args:
        filepath (str): 保存するパス
        scanImg (np.ndarray): 保存したい画像配列
    """
    img = min_max_normalize(scanImg)
    img = Image.fromarray(img * 255).convert("L")
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

    originalImgPath = f"./dataset/{setNumber}/in/sample.png"
    rightEyeImgPath = f"./dataset/{setNumber}/in/right_eye.png"
    leftEyeImgPath = f"./dataset/{setNumber}/in/left_eye.png"

    if normalize:
        originalImg = np.array(Image.open(originalImgPath))
        rightEyeImg = np.array(Image.open(rightEyeImgPath))
        leftEyeImg = np.array(Image.open(leftEyeImgPath))

        originalImg = (originalImg - originalImg.mean()) / originalImg.std()
        rightEyeImg = (rightEyeImg - rightEyeImg.mean()) / rightEyeImg.std()
        leftEyeImg = (leftEyeImg - leftEyeImg.mean()) / leftEyeImg.std()

        # (x - x.min()) / (x.max() - x.min())
        # originalImg = (originalImg - originalImg.min()) / (originalImg.max() - originalImg.min())
        # rightEyeImg = (rightEyeImg - rightEyeImg.min()) / (rightEyeImg.max() - rightEyeImg.min())
        # leftEyeImg = (leftEyeImg - leftEyeImg.min()) / (leftEyeImg.max() - leftEyeImg.min())

    else:
        # 画像読み込みつつndarrayに変換，asarray()を使うとread-onlyなデータができる．
        originalImg = np.asarray(Image.open(originalImgPath))
        rightEyeImg = np.asarray(Image.open(rightEyeImgPath))
        leftEyeImg = np.asarray(Image.open(leftEyeImgPath))

    imgDic = {"original": originalImg, "right_eye": rightEyeImg, "left_eye": leftEyeImg}

    return imgDic


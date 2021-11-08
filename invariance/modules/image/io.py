from PIL import Image
import numpy as np
from ..numeric import min_max_normalize

def save_image(filepath: str, scanImgArray: np.ndarray) -> None:
    """画像を保存する
    この関数内で正規化を行うので，元画像配列の状態は気にしなくてok

    Args:
        filepath (str): 保存するパス
        scanImgArray (np.ndarray): 保存したい画像配列
    """
    img = min_max_normalize(scanImgArray)
    img = Image.fromarray(img * 255).convert("L")
    # img.show()
    img.save(filepath)

def load_image(normalize: bool = True) -> dict:
    """画像を読み込む

    Args:
        normalize (bool, optional): 正規化するかしないか. Defaults to True.

    Returns:
        dict: 3種類の画像が格納されているDictionary
            key: 画像の種類(original, right_eye, left_eye)
            value: numpy.ndarray: 読み込んだ画像
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

        # (x - x.min()) / (x.max() - x.min())
        # originalImage = (originalImage - originalImage.min()) / (originalImage.max() - originalImage.min())
        # rightEyeImage = (rightEyeImage - rightEyeImage.min()) / (rightEyeImage.max() - rightEyeImage.min())
        # leftEyeImage = (leftEyeImage - leftEyeImage.min()) / (leftEyeImage.max() - leftEyeImage.min())

    else:
        # 画像読み込みつつndarrayに変換，asarray()を使うとread-onlyなデータができる．
        originalImage = np.asarray(Image.open(originalImagePath))
        rightEyeImage = np.asarray(Image.open(rightEyeImagePath))
        leftEyeImage = np.asarray(Image.open(leftEyeImagePath))

    imgDic = {"original": originalImage, "right_eye": rightEyeImage, "left_eye": leftEyeImage}

    return imgDic


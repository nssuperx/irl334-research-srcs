from PIL import Image
import numpy as np

def save_image(filepath: str, scanImgArray: np.ndarray) -> None:
    img = Image.fromarray(scanImgArray * 255).convert("L")
    # img.show()
    img.save(filepath)

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

from .core import ReceptiveField, CombinedReceptiveField
import numpy as np
from PIL import Image
from .numeric import min_max_normalize


def image_read_test(imgDic: dict) -> None:
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
        print(f"type:{str(type(value))} {str(key)} shape:{str(value.shape)}")

    for imgArray in imgDic.values():
        img = Image.fromarray(min_max_normalize(imgArray) * 255).convert("L")
        img.show()


def test_one_cRF(rightScanImg: np.ndarray, leftScanImg: np.ndarray,
                 rightTemplate: np.ndarray, leftTemplate: np.ndarray) -> None:
    rightRF = ReceptiveField((400, 700), rightScanImg, rightTemplate)
    leftRF = ReceptiveField((400, 740), leftScanImg, leftTemplate)
    combinedRF = CombinedReceptiveField(rightRF, leftRF)
    print("fci:" + str(combinedRF.get_fci()))

from .core import TemplateImage, ReceptiveField, CombinedReceptiveField
import numpy as np

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
        print("type:" + str(type(value)) + " " + str(key) + " shape:" + str(value.shape))

    """
    for imgArray in imgDic.values():
        img = Image.fromarray(imgArray)
        img.show()
    """

def test_one_cRF(rightScanImgArray: np.ndarray, leftScanImgArray: np.ndarray, rightTemplate: TemplateImage, leftTemplate: TemplateImage) -> None:
    rightRF = ReceptiveField((400,700), rightScanImgArray, rightTemplate)
    leftRF = ReceptiveField((400,740), leftScanImgArray, leftTemplate)
    combinedRF = CombinedReceptiveField(rightRF, leftRF)
    print("fci:" + str(combinedRF.get_fci()))


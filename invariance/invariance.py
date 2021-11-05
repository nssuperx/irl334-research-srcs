from typing import ContextManager, Dict, Tuple
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

class ReceptiveField:
    """
    走査後の出力画像から一番興奮してる座標取れそう．
    オリジナル画像の座標のみ持っておくべき．
    テンプレート画像を持ってないとfciが計算できないので，一応持っておく．
    参照渡しであることを信じる
    """
    # img: np.ndarray
    template: TemplateImage
    is_active: bool
    height: int
    width: int
    originalImgPos: Tuple[int, int]
    mostActivePos: Tuple[int, int]
    activity: float

    def __init__(self, originalImgPos: Tuple[int, int], scannedImgArray: np.ndarray, template: TemplateImage, height: int = 70, width: int = 70) -> None:
        # self.img = originalImg[originalImgPos]
        self.template = template
        self.originalImgPos = originalImgPos
        self.height = height
        self.width = width
        scannedArray = scannedImgArray[originalImgPos[0]:originalImgPos[0] + (height - template.img.shape[0]), originalImgPos[1]:originalImgPos[1] + (width - template.img.shape[1])]
        self.mostActivePos = np.unravel_index(np.argmax(scannedArray), scannedArray.shape)
        self.activity = np.max(scannedArray)

class CombinedReceptiveField:
    # TODO: ほんとはこっちでどのくらい重なるか調整できるべき
    rightRF: ReceptiveField
    leftRF: ReceptiveField
    height: int
    width: int
    overlap: int
    fci: float

    def __init__(self, rightRF: ReceptiveField, leftRF: ReceptiveField, height: int = 70, width: int = 110, overlap: int = 30) -> None:
        self.rightRF = rightRF
        self.leftRF = leftRF
        self.height = height
        self.width = width
        self.overlap = overlap
        
        # fci計算．関数化した方がいいかも
        noOverlap = (width - overlap) / 2
        x = int(noOverlap + self.leftRF.mostActivePos[0] - self.rightRF.mostActivePos[0])
        # y = abs(self.leftRF.mostActivePos[1] - self.rightRF.mostActivePos[1])
        rightOverlap: np.ndarray
        leftOverlap: np.ndarray
        if self.rightRF.mostActivePos[1] < self.leftRF.mostActivePos[1]:
            y = int(self.leftRF.mostActivePos[1] - self.rightRF.mostActivePos[1])
            rightOverlap = self.rightRF.template.img[x: , y: ]
            leftOverlap = self.leftRF.template.img[0:self.rightRF.template.img.shape[0] - x, 0:self.rightRF.template.img.shape[1] - y]
        else:
            y = int(self.rightRF.mostActivePos[1] - self.leftRF.mostActivePos[1])
            rightOverlap = self.rightRF.template.img[x: , 0:self.rightRF.template.img.shape[1] - y]
            leftOverlap = self.leftRF.template.img[0:self.rightRF.template.img.shape[0] - x, y: ]
        
        if rightOverlap.size == 0:
            self.fci = 0.0
        else:
            self.fci = np.sum(np.dot(rightOverlap, leftOverlap))
        

    def get_fci(self) -> float:
        return self.fci


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
    # rightEyeImgArray = imgDic["right_eye"]
    # leftEyeImgArray = imgDic["left_eye"]
    rightTemplate = TemplateImage(imgDic["right_eye"])
    leftTemplate = TemplateImage(imgDic["left_eye"])

    # 走査した画像配列を作成
    # rightScanImgArray = scan(originalImgArray, rightEyeImgArray)
    # leftScanImgArray = scan(originalImgArray, leftEyeImgArray)

    # 入力
    rightScanImgArray = np.load("./rightScanImgTmp.npy")
    leftScanImgArray = np.load("./leftScanImgTmp.npy")

    # 試しに一つReceptiveFieldを作る
    rightRF = ReceptiveField((400,700), rightScanImgArray, rightTemplate)
    leftRF = ReceptiveField((400,740), leftScanImgArray, leftTemplate)

    combinedRF = CombinedReceptiveField(rightRF, leftRF)

    print("fci:" + str(combinedRF.get_fci()))


    # np.save("./rightScanImgTmp", rightScanImgArray)
    image_save("./images/out/rightScanImg.png", rightScanImgArray)
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

if __name__ == "__main__":
    main()

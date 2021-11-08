from typing import Tuple
import numpy as np

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
        # NOTE: RFが担当する領域内のスキャン後の画像を切り抜いて，最も興奮している場所を探す
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
        self.calc_fci()

    # fci を計算する
    # NOTE: マイナスになってもok．抑制の入力．
    def calc_fci(self) -> float:
        noOverlap = (self.width - self.overlap) // 2
        if(self.rightRF.mostActivePos[1] + self.rightRF.template.img.shape[1] < noOverlap):
            self.fci = 0.0
            return self.fci

        if(self.leftRF.mostActivePos[1] > self.overlap):
            self.fci = 0.0
            return self.fci
        
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
            self.fci = np.sum(np.dot(rightOverlap.T, leftOverlap))
        
        return self.fci

    def get_fci(self) -> float:
        return self.fci
    # TODO: max fciが1になるように調整，でもそもそも重ならないならゼロなのでここの調整はどうすればいい？

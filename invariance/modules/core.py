from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw

from .numeric import zscore, min_max_normalize

class TemplateImage:
    def __init__(self, imgArray: np.ndarray) -> None:
        self.img: np.ndarray = imgArray
        self.mean: float = self.img.mean()
        self.variance: float = self.img.var()
        self.sd: float = self.img.std()

class ReceptiveField:
    """
    走査後の出力画像から一番興奮してる座標取れそう．
    オリジナル画像の座標のみ持っておくべき．
    テンプレート画像を持ってないとfciが計算できないので，一応持っておく．
    参照渡しであることを信じる
    """
    def __init__(self, originalImgPos: Tuple[int, int], scannedImgArray: np.ndarray, template: TemplateImage, height: int = 70, width: int = 70) -> None:
        self.template: TemplateImage = template
        self.originalImgPos: Tuple[int, int] = originalImgPos
        self.height: int = height
        self.width: int = width
        # NOTE: RFが担当する領域内のスキャン後の画像を切り抜いて，最も興奮している場所を探す
        scannedArray = scannedImgArray[originalImgPos[0]:originalImgPos[0] + (height - template.img.shape[0]), originalImgPos[1]:originalImgPos[1] + (width - template.img.shape[1])]
        self.mostActivePos: Tuple[int, int] = np.unravel_index(np.argmax(scannedArray), scannedArray.shape)
        self.activity: float = np.max(scannedArray)

    def show_img(self, originalImgArray: np.ndarray) -> None:
        """受容野が担当している，最も興奮している部分に枠を囲んだ画像を表示する

        Args:
            originalImgArray (np.ndarray): オリジナル画像配列
            オリジナル画像配列をこのクラスで持ちたくないので，表示するときのみスライスして使う．
        """
        im = Image.fromarray(min_max_normalize(originalImgArray[self.originalImgPos[0]:self.originalImgPos[0] + self.height, self.originalImgPos[1]:self.originalImgPos[1] + self.width]) * 255).convert('L')
        draw = ImageDraw.Draw(im)
        draw.rectangle((self.mostActivePos[1], self.mostActivePos[0], self.mostActivePos[1] + self.template.img.shape[1], self.mostActivePos[0] + self.template.img.shape[0]))
        im.show()

class CombinedReceptiveField:
    # TODO: ほんとはこっちでどのくらい重なるか調整できるべき
    def __init__(self, rightRF: ReceptiveField, leftRF: ReceptiveField, height: int = 70, width: int = 110, overlap: int = 30) -> None:
        self.rightRF: ReceptiveField = rightRF
        self.leftRF: ReceptiveField = leftRF
        self.height: int = height
        self.width: int = width
        self.overlap: int = overlap
        # self.fci: float = self.calc_fci()

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
        
        # TODO: overlap領域のサイズが違うときがある．完全にバグ．
        x = int(noOverlap + self.leftRF.mostActivePos[1] - self.rightRF.mostActivePos[1])
        # y = abs(self.leftRF.mostActivePos[1] - self.rightRF.mostActivePos[1])
        rightOverlap: np.ndarray
        leftOverlap: np.ndarray
        if self.rightRF.mostActivePos[0] < self.leftRF.mostActivePos[0]:
            y = int(self.leftRF.mostActivePos[0] - self.rightRF.mostActivePos[0])
            rightOverlap = self.rightRF.template.img[0:self.rightRF.template.img.shape[0] - y , x: ]
            leftOverlap = self.leftRF.template.img[y: , 0:self.rightRF.template.img.shape[1] - x]
        else:
            y = int(self.rightRF.mostActivePos[0] - self.leftRF.mostActivePos[0])
            rightOverlap = self.rightRF.template.img[y: , x: ]
            leftOverlap = self.leftRF.template.img[0:self.rightRF.template.img.shape[0] - y, x:self.rightRF.template.img.shape[1] - x]
        
        if rightOverlap.size == 0:
            self.fci = 0.0
        else:
            # 正規化
            rightOverlap = zscore(rightOverlap)
            leftOverlap = zscore(leftOverlap)
            self.fci = np.sum(np.dot(rightOverlap.T, leftOverlap))
        
        return self.fci

    def get_fci(self) -> float:
        return self.fci
    # NOTE: max fciが1になるように調整，でもそもそも重ならないならゼロなのでここの調整はどうすればいい？

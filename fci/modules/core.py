from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .numeric import min_max_normalize
from .vector2 import Vector2


class ReceptiveField:
    """
    走査後の出力画像から一番興奮してる座標取れそう．
    オリジナル画像の座標のみ持っておくべき．
    テンプレート画像を持ってないとfciが計算できないので，一応持っておく．
    参照渡しであることを信じる
    """

    def __init__(self, originalImgPos: Tuple[int, int], scannedImg: NDArray[np.float32],
                 template: NDArray[np.float32], height: int = 70, width: int = 70) -> None:
        self.template = template
        self.originalImgPos: Vector2 = Vector2(*originalImgPos)
        self.height: int = height
        self.width: int = width
        # NOTE: RFが担当する領域内のスキャン後の画像を切り抜いて，最も興奮している場所を探す
        oPos = self.originalImgPos                        # originalImgPos
        clipImg: NDArray[np.float32] = scannedImg[oPos.y:oPos.y + (height - template.shape[0]),
                                                  oPos.x:oPos.x + (width - template.shape[1])]
        self.mostActivePos: Vector2 = Vector2(*np.unravel_index(np.argmax(clipImg), clipImg.shape))
        self.activity = np.max(clipImg)

    def make_img(self, originalImg: NDArray[np.float32]) -> Image:
        """受容野が担当している，最も興奮している部分に枠を囲んだ画像を表示する

        Args:
            originalImg (NDArray[np.float32]): オリジナル画像配列
            オリジナル画像配列をこのクラスで持ちたくないので，表示するときのみスライスして使う．
        """
        oPos = self.originalImgPos                        # originalImgPos
        aPos = self.mostActivePos                         # mostActivePos
        tShape = Vector2(*self.template.shape)        # templateShape
        im = Image.fromarray(min_max_normalize(
            originalImg[oPos.y:oPos.y + self.height, oPos.x:oPos.x + self.width]) * 255).convert('L')
        draw = ImageDraw.Draw(im)
        draw.rectangle((aPos.x, aPos.y, aPos.x + tShape.x, aPos.y + tShape.y))
        return im


class CombinedReceptiveField:
    # TODO: ほんとはこっちでどのくらい重なるか調整できるべき
    def __init__(self, rightRF: ReceptiveField, leftRF: ReceptiveField,
                 height: int = 70, width: int = 110, overlap: int = 30) -> None:
        self.rightRF: ReceptiveField = rightRF
        self.leftRF: ReceptiveField = leftRF
        self.height: int = height
        self.width: int = width
        self.overlap: int = overlap
        self.fci: float = self.calc_fci()

    # fci を計算する
    # NOTE: マイナスになってもok．抑制の入力．
    def calc_fci(self) -> float:
        noOverlap = (self.width - self.overlap) // 2
        # right
        rMAPos: Vector2 = self.rightRF.mostActivePos
        rTShape: Vector2 = Vector2(*self.rightRF.template.shape)
        rlayer = np.zeros((self.height, self.width), dtype=np.float32)
        rmask = np.full_like(rlayer, False, dtype=bool)
        rlayer[rMAPos.y:rMAPos.y + rTShape.y, rMAPos.x:rMAPos.x + rTShape.x] = self.rightRF.template
        rmask[rMAPos.y:rMAPos.y + rTShape.y, rMAPos.x:rMAPos.x + rTShape.x] = True

        # left
        lMAPos: Vector2 = self.leftRF.mostActivePos
        lTShape: Vector2 = Vector2(*self.leftRF.template.shape)
        llayer = np.zeros((self.height, self.width), dtype=np.float32)
        lmask = np.full_like(llayer, False, dtype=bool)
        llayer[lMAPos.y:lMAPos.y + lTShape.y, lMAPos.x + noOverlap:lMAPos.x +
               lTShape.x + noOverlap] = self.leftRF.template
        lmask[lMAPos.y:lMAPos.y + lTShape.y, lMAPos.x + noOverlap:lMAPos.x + lTShape.x + noOverlap] = True

        mask = rmask * lmask
        maskedRlayer, maskedLlayer = rlayer[mask], llayer[mask]
        self.fci = np.sum(maskedRlayer * maskedLlayer)
        self.overlapPixels = maskedRlayer.size

        return self.fci

    def make_img(self, originalImg: NDArray[np.float32]) -> Image:
        oPos = self.rightRF.originalImgPos                           # originalImgPos
        lAPos = self.leftRF.mostActivePos                           # lightRFmostActivePos
        rAPos = self.rightRF.mostActivePos                          # rightRFmostActivePos
        lTShape = Vector2(*self.leftRF.template.shape)          # leftRFtemplateShape
        rTShape = Vector2(*self.rightRF.template.shape)         # rightRFtemplateShape
        noOverlap = (self.width - self.overlap) // 2
        img = Image.fromarray(min_max_normalize(
            originalImg[oPos.y:oPos.y + self.height, oPos.x:oPos.x + self.width]) * 255).convert('L')
        draw = ImageDraw.Draw(img)
        draw.rectangle((lAPos.x + noOverlap, lAPos.y, lAPos.x + lTShape.x + noOverlap, lAPos.y + lTShape.y))
        draw.rectangle((rAPos.x, rAPos.y, rAPos.x + rTShape.x, rAPos.y + rTShape.y), width=2)
        return img

    def get_fci(self) -> float:
        return self.fci

    def get_overlapPixels(self) -> int:
        # NOTE: 実験用なので，直接値を参照してもいい気がしている
        return self.overlapPixels

    # NOTE: max fciが1になるように調整，でもそもそも重ならないならゼロなのでここの調整はどうすればいい？

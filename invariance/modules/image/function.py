from typing import Tuple
import numpy as np
from ..numeric import zscore
from ..core import TemplateImage, ReceptiveField, CombinedReceptiveField
from ..vector2 import Vector2


def scan(originalImg: np.ndarray, template: TemplateImage) -> np.ndarray:
    # 走査
    # TODO: 遅すぎるのでなんとかする
    oShape = Vector2(*originalImg.shape)
    tShape = Vector2(*template.img.shape)
    scanImg = np.empty((oShape.y - tShape.y, oShape.x - tShape.x), dtype=np.float64)
    for y in range(scanImg.shape[0]):
        for x in range(scanImg.shape[1]):
            # 相関係数出す範囲をスライス
            scanTargetImg = originalImg[y:y + tShape.y, x:x + tShape.x]
            # 正規化
            scanTargetImg = zscore(scanTargetImg)
            # scanImg[y][x] = corrcoef(scanTargetImg, template.img)
            # scanImg[y][x] = corrcoef_template(scanTargetImg, template)
            # NOTE: 平均0 分散1なので，以下でもok．
            scanImg[y][x] = np.mean(scanTargetImg * template.img)

    return scanImg


def scan_combinedRF(cRFHeight: int, cRFWidth: int, RFheight: int, RFWidth: int, scanStep: int, originalImg: np.ndarray,
                    rightScanImg: np.ndarray, rightTemplate: TemplateImage,
                    leftScanImg: np.ndarray, leftTemplate: TemplateImage) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像全体のfciを計算する

    Args:
        cRFHeight (int): combined ReceptiveField の縦の長さ
        cRFWidth (int): combined ReceptiveField の横の長さ
        scanStep (int): スキャンするときのステップ数（スキャンをとばす幅）
        originalImg (np.ndarray): 元画像
        rightScanImg (np.ndarray): 右目でスキャンした相関係数配列
        rightTemplate (TemplateImage): 右目テンプレート
        leftScanImg (np.ndarray): 左目でスキャンした相関係数配列
        leftTemplate (TemplateImage): 左目テンプレート

    Returns:
        np.ndarray: fci配列
    """
    oShape = Vector2(*originalImg.shape)
    fci = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)
    rr = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)
    lr = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)
    for y in range(0, oShape.y - cRFHeight, scanStep):
        for x in range(0, oShape.x - cRFWidth, scanStep):
            rightRF = ReceptiveField((y, x), rightScanImg, rightTemplate, RFheight, RFWidth)
            leftRF = ReceptiveField((y, x + (cRFWidth - RFWidth)), leftScanImg, leftTemplate, RFheight, RFWidth)
            combinedRF = CombinedReceptiveField(rightRF, leftRF, cRFHeight, cRFWidth, (RFWidth * 2 - cRFWidth))
            # combinedRF.save_img(originalImg, f'./imgout/y{y:04}x{x:04}.png')
            fci[y//scanStep][x//scanStep] = combinedRF.get_fci()
            rr[y//scanStep][x//scanStep] = combinedRF.rightRF.activity
            lr[y//scanStep][x//scanStep] = combinedRF.leftRF.activity

    # fci = min_max_normalize(fci)
    return fci, rr, lr

import numpy as np
from ..numeric import min_max_normalize, zscore, calc_corrcoef
from ..core import TemplateImage, ReceptiveField, CombinedReceptiveField
from ..vector2 import Vector2

def scan(originalImg: np.ndarray, templateImg: np.ndarray) -> np.ndarray:
    # 走査
    # TODO: 遅すぎるのでなんとかする
    oShape = Vector2(*originalImg.shape)
    tShape = Vector2(*templateImg.shape)
    scanImg = np.empty((oShape.y - tShape.y, oShape.x - tShape.x))
    for y in range(scanImg.shape[0]):
        for x in range(scanImg.shape[1]):
            # 相関係数出す範囲をスライス
            scanTargetImg = originalImg[y:y+tShape.y, x:x+tShape.x]
            # 正規化
            scanTargetImg = zscore(scanTargetImg)
            # cov = np.mean(np.multiply(scanTargetImg, templateImg))
            # cov = np.mean(np.multiply(scanTargetImg, templateImg)) - scanTargetImg.mean() * templateImg.mean()
            # scanImg[y][x] = np.corrcoef(scanTargetImg.flatten(), templateImg.flatten())[0][1]
            # scanImg[y][x] = cov / (scanTargetImg.std() * templateImg.std())
            scanImg[y][x] = calc_corrcoef(scanTargetImg, templateImg)

    return scanImg

def scan_combinedRF(cRFHeight: int, cRFWidth: int, RFheight: int, RFWidth: int, scanStep: int, originalImg: np.ndarray,
                    rightScanImg: np.ndarray, rightTemplate: TemplateImage,
                    leftScanImg: np.ndarray, leftTemplate: TemplateImage) -> np.ndarray:
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
    fci = np.zeros(((originalImg.shape[0] - cRFHeight) // scanStep, (originalImg.shape[1] - cRFWidth) // scanStep))
    for y in range(0, originalImg.shape[0] - cRFHeight, scanStep):
        for x in range(0, originalImg.shape[1] - cRFWidth, scanStep):
            rightRF = ReceptiveField((y, x), rightScanImg, rightTemplate, RFheight, RFWidth)
            leftRF = ReceptiveField((y, x + (cRFWidth - RFWidth)), leftScanImg, leftTemplate, RFheight, RFWidth)
            combinedRF = CombinedReceptiveField(rightRF, leftRF, cRFHeight, cRFWidth, (RFWidth * 2 - cRFWidth))
            combinedRF.save_img(originalImg, f'./imgout/y{y:03}x{x:03}.png')
            # combinedRF.show_img(originalImg)
            # fci[y//scanStep][x//scanStep] = combinedRF.get_fci()

    # fci = min_max_normalize(fci)
    return fci

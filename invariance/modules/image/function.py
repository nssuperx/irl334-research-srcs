import numpy as np
from ..numeric import min_max_normalize
from ..core import TemplateImage, ReceptiveField, CombinedReceptiveField

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

def scan_combinedRF(cRFHeight: int, cRFWidth: int, scanStep: int, originalImgArray: np.ndarray,
                    rightScanImgArray: np.ndarray, rightTemplate: TemplateImage,
                    leftScanImgArray: np.ndarray, leftTemplate: TemplateImage) -> np.ndarray:
    """画像全体のfciを計算する

    Args:
        cRFHeight (int): combined ReceptiveField の縦の長さ
        cRFWidth (int): combined ReceptiveField の横の長さ
        scanStep (int): スキャンするときのステップ数（スキャンをとばす幅）
        originalImgArray (np.ndarray): 元画像
        rightScanImgArray (np.ndarray): 右目でスキャンした相関係数配列
        rightTemplate (TemplateImage): 右目テンプレート
        leftScanImgArray (np.ndarray): 左目でスキャンした相関係数配列
        leftTemplate (TemplateImage): 左目テンプレート

    Returns:
        np.ndarray: fci配列
    """
    fciArray = np.zeros(((originalImgArray.shape[0] - cRFHeight) // scanStep, (originalImgArray.shape[1] - cRFWidth) // scanStep))
    for y in range(0, originalImgArray.shape[0] - cRFHeight, scanStep):
        for x in range(0, originalImgArray.shape[1] - cRFWidth, scanStep):
            rightRF = ReceptiveField((y, x), rightScanImgArray, rightTemplate)
            leftRF = ReceptiveField((y, x + (cRFWidth- cRFHeight)), leftScanImgArray, leftTemplate)
            combinedRF = CombinedReceptiveField(rightRF, leftRF)
            fciArray[y//scanStep][x//scanStep] = combinedRF.get_fci()

    # fciArray = min_max_normalize(fciArray)
    return fciArray

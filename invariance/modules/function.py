import numpy as np
from PIL import Image
from tqdm import tqdm

from .io import FciDataManager
from .numeric import zscore
from .core import ReceptiveField, CombinedReceptiveField
from .vector2 import Vector2
from .results import FciResultBlock


def scan(originalImg: np.ndarray, template: np.ndarray) -> np.ndarray:
    """（非推奨）
    テンプレートマッチングする．正規化相関係数．
    NOTE: 遅すぎるので使わない．

    Args:
        originalImg (np.ndarray): 元の画像
        template (np.ndarray): テンプレート画像

    Returns:
        np.ndarray: テンプレートマッチング後のnumpy配列
    """
    oShape = Vector2(*originalImg.shape)
    tShape = Vector2(*template.shape)
    scanImg = np.empty((oShape.y - tShape.y + 1, oShape.x - tShape.x + 1), dtype=np.float32)
    for y in tqdm(range(scanImg.shape[0])):
        for x in range(scanImg.shape[1]):
            # 相関係数出す範囲をスライス
            scanTargetImg = originalImg[y:y + tShape.y, x:x + tShape.x]
            # 正規化
            scanTargetImg = zscore(scanTargetImg)
            # scanImg[y][x] = corrcoef(scanTargetImg, template)
            # NOTE: 平均0 分散1なので，以下でもok．
            scanImg[y][x] = np.mean(scanTargetImg * template)

    return scanImg


def scan_combinedRF(cRFHeight: int, cRFWidth: int, RFheight: int, RFWidth: int, scanStep: int, originalImg: np.ndarray,
                    rightScanImg: np.ndarray, rightTemplate: np.ndarray,
                    leftScanImg: np.ndarray, leftTemplate: np.ndarray,
                    dataMgr: FciDataManager, saveImage: bool = False) -> FciResultBlock:
    """画像全体のfciを計算する

    Args:
        cRFHeight (int): combined ReceptiveField の縦の長さ
        cRFWidth (int): combined ReceptiveField の横の長さ
        scanStep (int): スキャンするときのステップ数（スキャンをとばす幅）
        originalImg (np.ndarray): 元画像
        rightScanImg (np.ndarray): 右目でスキャンした相関係数配列
        rightTemplate (np.ndarray): 右目テンプレート
        leftScanImg (np.ndarray): 左目でスキャンした相関係数配列
        leftTemplate (np.ndarray): 左目テンプレート

    Returns:
        NOTE: 後で変更される可能性あり
        FciResultBlock: 結果をまとめたもの
    """

    # TODO: 冗長なのでなんとかする
    # TODO: 並列処理はどうする？
    oShape = Vector2(*originalImg.shape)
    fci = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float32)
    rr = np.empty_like(fci, dtype=np.float32)
    lr = np.empty_like(fci, dtype=np.float32)
    raposy = np.empty_like(fci, dtype=np.uint32)
    raposx = np.empty_like(fci, dtype=np.uint32)
    laposy = np.empty_like(fci, dtype=np.uint32)
    laposx = np.empty_like(fci, dtype=np.uint32)
    overlapPixels = np.empty_like(fci, dtype=np.uint32)

    crfx = np.tile(np.arange(fci.shape[1], dtype=np.uint32) * scanStep, (fci.shape[0], 1))
    crfy = np.tile(np.arange(fci.shape[0], dtype=np.uint32) * scanStep, (fci.shape[1], 1)).T

    for y in tqdm(range(0, fci.shape[0])):
        for x in range(0, fci.shape[1]):
            rightRF = ReceptiveField((y * scanStep, x * scanStep), rightScanImg, rightTemplate, RFheight, RFWidth)
            leftRF = ReceptiveField((y * scanStep, x * scanStep + (cRFWidth - RFWidth)),
                                    leftScanImg, leftTemplate, RFheight, RFWidth)
            cRF = CombinedReceptiveField(rightRF, leftRF, cRFHeight, cRFWidth, (RFWidth * 2 - cRFWidth))
            if saveImage:
                img: Image = cRF.make_img(originalImg)
                img.save(f"{dataMgr.out_dirpath}/crf/y{y*scanStep:04}x{x*scanStep:04}.png")
            fci[y][x] = cRF.get_fci()
            overlapPixels[y][x] = cRF.get_overlapPixels()
            rr[y][x] = cRF.rightRF.activity
            lr[y][x] = cRF.leftRF.activity
            rpos = cRF.rightRF.originalImgPos + cRF.rightRF.mostActivePos
            lpos = cRF.leftRF.originalImgPos + cRF.leftRF.mostActivePos
            raposy[y][x], raposx[y][x] = rpos.y, rpos.x
            laposy[y][x], laposx[y][x] = lpos.y, lpos.x

    return FciResultBlock(crfy, crfx, fci, rr, lr, raposy, raposx, laposy, laposx, overlapPixels)

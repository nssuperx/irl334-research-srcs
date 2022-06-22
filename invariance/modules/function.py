import numpy as np

from .io import FciDataManager
from .numeric import zscore
from .core import TemplateImage, ReceptiveField, CombinedReceptiveField
from .vector2 import Vector2
from .results import FciResultBlock
from tqdm import tqdm


def scan(originalImg: np.ndarray, template: TemplateImage) -> np.ndarray:
    # 走査
    # TODO: 遅すぎるのでなんとかする
    oShape = Vector2(*originalImg.shape)
    tShape = Vector2(*template.img.shape)
    scanImg = np.empty((oShape.y - tShape.y, oShape.x - tShape.x), dtype=np.float64)
    for y in tqdm(range(scanImg.shape[0])):
        for x in tqdm(range(scanImg.shape[1]), leave=False):
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
                    leftScanImg: np.ndarray, leftTemplate: TemplateImage,
                    dataMgr: FciDataManager, saveImage: bool = False) -> FciResultBlock:
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
        NOTE: 後で変更される可能性あり
        FciResultBlock: 結果をまとめたもの
    """

    # TODO: 冗長なのでなんとかする
    # TODO: 並列処理はどうする？
    oShape = Vector2(*originalImg.shape)
    fci = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)
    rr = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)
    lr = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)
    raposy = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.uint32)
    raposx = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.uint32)
    laposy = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.uint32)
    laposx = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.uint32)

    for y in tqdm(range(0, fci.shape[0])):
        for x in range(0, fci.shape[1]):
            rightRF = ReceptiveField((y * scanStep, x * scanStep), rightScanImg, rightTemplate, RFheight, RFWidth)
            leftRF = ReceptiveField((y * scanStep, x * scanStep + (cRFWidth - RFWidth)),
                                    leftScanImg, leftTemplate, RFheight, RFWidth)
            cRF = CombinedReceptiveField(rightRF, leftRF, cRFHeight, cRFWidth, (RFWidth * 2 - cRFWidth))
            if saveImage:
                cRF.save_img(originalImg, f"{dataMgr.get_out_dirpath()}/crf/y{y*scanStep:04}x{x*scanStep:04}.png")
            fci[y][x] = cRF.get_fci()
            rr[y][x] = cRF.rightRF.activity
            lr[y][x] = cRF.leftRF.activity
            rpos = cRF.rightRF.originalImgPos + cRF.rightRF.mostActivePos
            lpos = cRF.leftRF.originalImgPos + cRF.leftRF.mostActivePos
            raposy[y][x], raposx[y][x] = rpos.y, rpos.x
            laposy[y][x], laposx[y][x] = lpos.y, lpos.x

    # まだできてない
    crfx = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)
    crfy = np.empty(((oShape.y - cRFHeight) // scanStep, (oShape.x - cRFWidth) // scanStep), dtype=np.float64)

    return FciResultBlock(crfy, crfx, fci, rr, lr, raposy, raposx, laposy, laposx)

import os
import json
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from .numeric import min_max_normalize, zscore


class FciDataManager:
    def __init__(self, datasetNumber: int, setting: str = "default") -> None:
        self.datasetNumber = datasetNumber
        self.setting = setting
        os.makedirs(f"./dataset/{self.datasetNumber}/in", exist_ok=True)
        os.makedirs(f"./dataset/{self.datasetNumber}/in/color", exist_ok=True)
        os.makedirs(f"./dataset/{self.datasetNumber}/array/{self.setting}", exist_ok=True)
        os.makedirs(f"./dataset/{self.datasetNumber}/out/{self.setting}/crf", exist_ok=True)
        os.makedirs(f"./dataset/{self.datasetNumber}/out/{self.setting}/crf-clean", exist_ok=True)

        with open(f"./dataset/{self.datasetNumber}/setting.json", "r") as f:
            jsondata = json.load(f)

        self.params = jsondata[setting]

    def load_image(self, normalize: bool = True) -> None:
        """画像を読み込む

        Args:
            normalize (bool, optional): 正規化するかしないか. Defaults to True.
        """

        originalImgPath = f"./dataset/{self.datasetNumber}/in/original.png"
        rightTemplatePath = f"./dataset/{self.datasetNumber}/in/{self.params['right_template']}"
        leftTemplatePath = f"./dataset/{self.datasetNumber}/in/{self.params['left_template']}"

        if normalize:
            self.originalImg = zscore(np.array(Image.open(originalImgPath), dtype=np.float32))
            self.rightTemplate = zscore(np.array(Image.open(rightTemplatePath), dtype=np.float32))
            self.leftTemplate = zscore(np.array(Image.open(leftTemplatePath), dtype=np.float32))

        else:
            self.originalImg = np.array(Image.open(originalImgPath), dtype=np.float32)
            self.rightTemplate = np.array(Image.open(rightTemplatePath), dtype=np.float32)
            self.leftTemplate = np.array(Image.open(leftTemplatePath), dtype=np.float32)

    def load_scan_array(self) -> None:
        self.rightScanImg: NDArray[np.number] = np.load(
            f"./dataset/{self.datasetNumber}/array/{self.setting}/rightScanImg.npy")
        self.leftScanImg: NDArray[np.number] = np.load(
            f"./dataset/{self.datasetNumber}/array/{self.setting}/leftScanImg.npy")

    def save_scan_image(self, rightScanImg: NDArray[np.number],
                        leftScanImg: NDArray[np.number], dirpath: str = None) -> None:
        if dirpath is None:
            dirpath = f"./dataset/{self.datasetNumber}/out/{self.setting}"
        rightimg = Image.fromarray(min_max_normalize(rightScanImg) * 255).convert("L")
        leftimg = Image.fromarray(min_max_normalize(leftScanImg) * 255).convert("L")
        rightimg.save(f"{dirpath}/rightScanImg.png")
        leftimg.save(f"{dirpath}/leftScanImg.png")

    def save_scan_array(self, rightScanImg: NDArray[np.uint32],
                        leftScanImg: NDArray[np.uint32], dirpath: str = None) -> None:
        if dirpath is None:
            dirpath = f"./dataset/{self.datasetNumber}/array/{self.setting}"
        np.save(f"{dirpath}/rightScanImg", rightScanImg)
        np.save(f"{dirpath}/leftScanImg", leftScanImg)

    @property
    def out_dirpath(self) -> str:
        """ファイルを保存してほしいディレクトリを返す
        TODO: いい設計思いつくまでこれで妥協

        Returns:
            str: ファイルの保存先ディレクトリ
        """
        return f"./dataset/{self.datasetNumber}/out/{self.setting}"

    @property
    def dirpath(self) -> str:
        """データセットのルートのパスを返す
        Returns:
            str: データセットのパス
        """
        return f"./dataset/{self.datasetNumber}"


def save_image(filepath: str, scanImg: NDArray[np.float32], heat: bool = False) -> None:
    """
    画像を保存する
    この関数内で正規化を行うので，元画像配列の状態は気にしなくてok

    Args:
        filepath (str): 保存するパス
        scanImg (NDArray[np.float32]): 保存したい画像配列
        heat (bool): 出力画像をヒートマップ風にするフラグ
    """

    scanImg = min_max_normalize(scanImg) * 255
    scanImg = scanImg.astype(np.uint8)

    if heat:
        mat = np.stack([scanImg, np.full_like(scanImg, 255), np.full_like(scanImg, 255)], axis=2)
        img = Image.fromarray(mat, "HSV").convert("RGB")
    else:
        img = Image.fromarray(scanImg).convert("L")

    img.save(filepath)

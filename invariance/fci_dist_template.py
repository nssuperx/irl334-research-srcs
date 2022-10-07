import sys
import numpy as np
import matplotlib.pyplot as plt

from modules.io import FciDataManager
from modules.vector2 import Vector2


default_dataset = 1
args = sys.argv


def main():
    if (len(args) >= 2):
        dataset_number = int(args[1])
    else:
        dataset_number = default_dataset
    datas: FciDataManager = FciDataManager(dataset_number)
    datas.load_image()
    rt = datas.rightTemplate
    lt = datas.leftTemplate
    rtshape = Vector2(*rt.shape)
    ltshape = Vector2(*lt.shape)

    # 右テンプレートを中央に配置した配列と，そのマスク配列を作る
    # 配列の大きさ: right + (left - 1) * 2 == left * 2 + right - 2
    array_size = ltshape * Vector2(2, 2) + rtshape - Vector2(2, 2)
    rlayer = np.zeros((array_size.y, array_size.x), dtype=np.float32)
    rlayer[ltshape.y - 1:ltshape.y - 1 + rtshape.y, ltshape.x - 1:ltshape.x - 1 + rtshape.x] = rt
    rmask = np.full_like(rlayer, False, dtype=bool)
    rmask[ltshape.y - 1:ltshape.y - 1 + rtshape.y, ltshape.x - 1:ltshape.x - 1 + rtshape.x] = True

    fci_dist = []
    fci_pixels = []
    for y in range(array_size.y - ltshape.y + 1):
        for x in range(array_size.x - ltshape.x + 1):
            llayer = np.zeros((array_size.y, array_size.x), dtype=np.float32)
            llayer[y:y + ltshape.y, x:x + ltshape.x] = lt
            lmask = np.full_like(llayer, False, dtype=bool)
            lmask[y:y + ltshape.y, x:x + ltshape.x] = True
            mask = rmask * lmask
            maskedPixels = rlayer[mask] * llayer[mask]
            # fci_dist.append(np.sum(rlayer * llayer))
            fci_dist.append(np.sum(maskedPixels))
            fci_pixels.append(maskedPixels.size)

    print(f"right shape: {rt.shape}, left shape: {lt.shape}")
    print(f"right pixels: {rt.size}, left pixels: {lt.size}")
    print(f"fci num: {len(fci_dist)}")

    hist_bins = 100
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(fci_dist, bins=hist_bins)

    # 特定の値のビンに色つけたいとき
    # _, _, patches = ax.hist(fci_dist, bins=hist_bins)
    # watch_value: float = 163.71954345703125
    # watch_point: int = int(linear_map(watch_value, min(fci_dist), max(fci_dist), 0, hist_bins - 1))
    # patches[watch_point].set_facecolor('orange')

    ax.set_xlabel("raw fci")
    ax.set_title("fci histogram (template image)")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(fci_pixels, fci_dist)
    ax.set_ylabel("raw fci")
    ax.set_xlabel("overlap pixel")
    plt.show()


# 参考: https://www.arduino.cc/reference/en/language/functions/math/map/
def linear_map(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if __name__ == "__main__":
    main()

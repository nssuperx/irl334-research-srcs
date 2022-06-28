import numpy as np
import matplotlib.pyplot as plt

from modules.io import FciDataManager
from modules.vector2 import Vector2


def main() -> None:
    datas = FciDataManager(1)
    datas.load_image()
    rt = datas.rightEyeImg
    lt = datas.leftEyeImg
    rtshape = Vector2(*rt.shape)
    ltshape = Vector2(*lt.shape)

    rlayer = np.zeros((rtshape.y + ltshape.y, rtshape.x + ltshape.x), dtype=np.float32)
    rlayer[:rtshape.y, :rtshape.x] = rt
    rmask = np.full_like(rlayer, False, dtype=bool)
    rmask[:rtshape.y, :rtshape.x] = True

    fciDist = []
    fciPixles = []
    for y in range(rtshape.y):
        for x in range(rtshape.x):
            llayer = np.zeros((rtshape.y + ltshape.y, rtshape.x + ltshape.x), dtype=np.float32)
            llayer[y:y + ltshape.y, x:x + ltshape.x] = lt
            lmask = np.full_like(llayer, False, dtype=bool)
            lmask[y:y + ltshape.y, x:x + ltshape.x] = True
            mask = rmask * lmask
            maskedPixels = rlayer[mask] * llayer[mask]
            # fciDist.append(np.sum(rlayer * llayer))
            fciDist.append(np.sum(maskedPixels))
            fciPixles.append(maskedPixels.size)

    print(f"right shape: {rt.shape}, left shape: {lt.shape}")
    print(f"right pixels: {rt.size}, left pixels: {lt.size}")
    print(f"fci num: {len(fciDist)}")

    hist_bins = 100
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    _, _, patches = ax.hist(fciDist, bins=hist_bins)
    watch_value: float = 163.719517963921
    watch_point: int = int(watch_value * hist_bins / max(fciDist))
    patches[watch_point].set_facecolor('orange')
    ax.set_xlabel("raw fci")
    ax.set_title("fci histogram (template image)")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(fciPixles, fciDist)
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

from modules.image.io import FciDataManager
from modules.vector2 import Vector2


def main() -> None:
    datas = FciDataManager(3)
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

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(fciDist, bins=100)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(fciPixles, fciDist)
    plt.show()


if __name__ == "__main__":
    main()

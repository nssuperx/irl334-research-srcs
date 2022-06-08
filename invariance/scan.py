import numpy as np

from modules.core import TemplateImage
from modules.image.io import load_image, save_image
from modules.image.function import scan


def main():
    for i in range(1, 3 + 1):
        imgDic = load_image(i)
        # image_read_test(imgDic)
        originalImgArray = imgDic["original"]
        rightTemplate = TemplateImage(imgDic["right_eye"])
        leftTemplate = TemplateImage(imgDic["left_eye"])

        # 走査した画像配列を作成
        rightScanImgArray = scan(originalImgArray, rightTemplate)
        leftScanImgArray = scan(originalImgArray, leftTemplate)
        save_image(f"./dataset/{i}/out/rightScanImg.png", rightScanImgArray)
        save_image(f"./dataset/{i}/out/leftScanImg.png", leftScanImgArray)

        np.save(f"./dataset/{i}/array/rightScanImg", rightScanImgArray)
        np.save(f"./dataset/{i}/array/leftScanImg", leftScanImgArray)


if __name__ == "__main__":
    main()

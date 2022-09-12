import sys
import cv2

from modules.io import FciDataManager

default_dataset = 1
args = sys.argv


def main():
    if (len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset

    data = FciDataManager(dataset_number)
    data.load_image()
    originalImg = data.originalImg
    rightTemplate = data.rightTemplate
    leftTemplate = data.leftTemplate

    # 走査した画像配列を作成
    rightScanImg = cv2.matchTemplate(originalImg, rightTemplate, method=cv2.TM_CCOEFF_NORMED)
    leftScanImg = cv2.matchTemplate(originalImg, leftTemplate, method=cv2.TM_CCOEFF_NORMED)

    # 保存
    data.save_scan_image(rightScanImg, leftScanImg)
    data.save_scan_array(rightScanImg, leftScanImg)


if __name__ == "__main__":
    main()

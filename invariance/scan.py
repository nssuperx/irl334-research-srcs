import sys

from modules.core import TemplateImage
from modules.io import FciDataManager
from modules.function import scan

default_dataset = 1
args = sys.argv


def main():
    if(len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset

    data = FciDataManager(dataset_number)
    data.load_image()
    originalImg = data.originalImg
    rightTemplate = TemplateImage(data.rightEyeImg)
    leftTemplate = TemplateImage(data.leftEyeImg)

    # 走査した画像配列を作成
    rightScanImg = scan(originalImg, rightTemplate)
    leftScanImg = scan(originalImg, leftTemplate)

    # 保存
    data.save_scan_image(rightScanImg, leftScanImg)
    data.save_scan_array(rightScanImg, leftScanImg)


if __name__ == "__main__":
    main()

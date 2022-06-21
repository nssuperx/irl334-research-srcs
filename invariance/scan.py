from modules.core import TemplateImage
from modules.io import FciDataManager
from modules.function import scan


def main():
    for i in range(1, 4 + 1):
        data = FciDataManager(i)
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

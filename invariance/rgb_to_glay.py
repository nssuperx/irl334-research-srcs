import sys
from PIL import Image

from modules.io import FciDataManager

default_dataset = 5
args = sys.argv


def main():
    if(len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset
    data: FciDataManager = FciDataManager(dataset_number)
    inoutFilePaths = ((f"{data.get_dirpath()}/in/color/original.png", f"{data.get_dirpath()}/in/original.png"),
                      (f"{data.get_dirpath()}/in/color/right_eye.png", f"{data.get_dirpath()}/in/right_eye.png"),
                      (f"{data.get_dirpath()}/in/color/left_eye.png", f"{data.get_dirpath()}/in/left_eye.png"))

    for pair in inoutFilePaths:
        img = Image.open(pair[0])
        gray = img.convert("L")
        gray.save(pair[1])


if __name__ == "__main__":
    main()

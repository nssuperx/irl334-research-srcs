from os import sched_get_priority_max
from PIL import Image

def main():
    inoutPairfilePaths = (("./images/original/sample.png", "./images/sample.png"),
        ("./images/original/right_eye.png", "./images/right_eye.png"),
        ("./images/original/left_eye.png", "./images/left_eye.png"))
    
    for pair in inoutPairfilePaths:
        img = Image.open(pair[0])
        gray = img.convert("L")
        gray.save(pair[1])

if __name__ == "__main__":
    main()

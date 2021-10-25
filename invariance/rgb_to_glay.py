from PIL import Image

def main():
    inoutPairfilePaths = (("./images/original/sample_small.png", "./images/in/sample.png"),
        ("./images/original/right_eye_small.png", "./images/in/right_eye.png"),
        ("./images/original/left_eye_small.png", "./images/in/left_eye.png"))
    
    for pair in inoutPairfilePaths:
        img = Image.open(pair[0])
        gray = img.convert("L")
        gray.save(pair[1])

if __name__ == "__main__":
    main()

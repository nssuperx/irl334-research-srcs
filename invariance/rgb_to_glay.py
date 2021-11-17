from PIL import Image

def main():
    for i in range(1, 3+1):
        inoutPairfilePaths = (("./dataset/" + str(i) + "/in/sample.png", "./dataset/" + str(i) + "/in/sample.png"),
            ("./dataset/" + str(i) + "/in/right_eye.png", "./dataset/" + str(i) + "/in/right_eye.png"),
            ("./dataset/" + str(i) + "/in/left_eye.png", "./dataset/" + str(i) + "/in/left_eye.png"))
        
        for pair in inoutPairfilePaths:
            img = Image.open(pair[0])
            gray = img.convert("L")
            gray.save(pair[1])

    '''
    1枚

    img = Image.open("./images/set/3/in/original.jpg")
    gray = img.convert("L")
    gray.save("./images/set/3/in/sample.png")
    '''


if __name__ == "__main__":
    main()

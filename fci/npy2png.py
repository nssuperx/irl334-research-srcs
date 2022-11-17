import numpy as np
from PIL import Image

test = np.load("./dataset/1/original-arrays/original.npy")
img = Image.fromarray(test).convert("L")
img.save("./original.png")

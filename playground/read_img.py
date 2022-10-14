from PIL import Image
import numpy as np

img = Image.open("./tmp.png")

arr = np.array(img)

d = Image.fromarray(arr)
d.show()

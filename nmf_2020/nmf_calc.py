import random
import math
import numpy as np
import matplotlib.pyplot as plt

from modules.nmf import NMF
from modules.data_function import setup_mnist
from modules.array_function import make_baseGridImage, normalization

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

m = 1000       # 画像数
r = 100         # 基底数
iteration = 50

def main():
    V, labels = setup_mnist(image_num=m)

    print("V shape:" + str(V.shape))

    nmf = NMF()
    nmf.calc(V, r, iteration)
    W = nmf.W
    H = nmf.H
    F_list = nmf.loss_LOG

    plt.plot(range(iteration), F_list)
    plt.show()

    # 確認!!
    sample_index = np.random.randint(0,m)
    print(labels[sample_index])
    fig = plt.figure()
    original_npimg = V[:,sample_index].reshape((28, 28))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(original_npimg, cmap="Greys_r")
    
    restore_npimg = np.dot(W,H[:,sample_index]).reshape((28, 28))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(restore_npimg, cmap="Greys_r")
    plt.show()

    # 基底画像の確認
    W_image = W.T.reshape(r, 28, 28)
    W_images = make_baseGridImage(W_image, r, normalize=True)
    img_max_abs = np.abs(W_image.max())
    img_min_abs = np.abs(W_image.min())
    colormap_range = (img_max_abs if img_max_abs > img_min_abs else img_min_abs) / 2
    sqrt_n = 28
    sqrt_r = int(np.sqrt(r))
    plt.imshow(W_images, cmap="Greys", extent=(0, sqrt_n * sqrt_r, 0, sqrt_n * sqrt_r))
    plt.xticks(range(0, sqrt_n * sqrt_r, sqrt_n))
    plt.yticks(range(0, sqrt_n * sqrt_r, sqrt_n))
    plt.grid(which = "major", color = "black", alpha = 0.8, linestyle = "--", linewidth = 1)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.show()

if __name__ == "__main__":
    main()
    
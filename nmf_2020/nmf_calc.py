import random
import math
import numpy as np
import matplotlib.pyplot as plt

from modules.nmf import NMF
from modules.data import setup_mnist
from modules.array import make_baseGridImage, normalization
from modules.visualize import show_base_glid, show_reconstruct_pairs, show_base_weight

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

m = 1000       # 画像数
r = 100         # 基底数
iteration = 100

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

    reconstruct_V = np.dot(W, H)

    show_base_glid(W, r, img_cmap="Greys")
    show_reconstruct_pairs(V, reconstruct_V, m)
    # show_base_weight(V, reconstruct_V, W, H, r, m)


if __name__ == "__main__":
    main()
    
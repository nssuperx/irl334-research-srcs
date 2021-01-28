import random
import math
import numpy as np
import matplotlib.pyplot as plt

from modules.nmf import NMF
from modules.data import setup_mnist, csv_out
from modules.array import make_baseGridImage, normalization
from modules.visualize import show_base_grid, show_reconstruct_pairs, show_base_weight, show_graph

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

m = 1000       # 画像数
# r = 1000        # 基底数
r_list = [25, 49, 100, 784, 1000]
iteration = 200

def main():
    V, labels = setup_mnist(image_num=m)

    print("V shape:" + str(V.shape))

    for r in r_list:
        nmf = NMF()
        nmf.calc(V, r, iteration)
        W = nmf.W
        H = nmf.H
        F_list = nmf.loss_LOG
        csv_out('nmf_r' + str(r) + '.csv', ('iteration', 'F'), (range(1, iteration + 1), F_list))

    # show_graph(range(iteration), F_list, 'iteration', 'F')

    '''
    reconstruct_V = np.dot(W, H)

    show_base_grid(W, r, img_cmap="Greens", img_normalize=True)
    show_reconstruct_pairs(V, reconstruct_V, m)
    # show_base_weight(V, reconstruct_V, W, H, r, m)
    '''



if __name__ == "__main__":
    main()
    
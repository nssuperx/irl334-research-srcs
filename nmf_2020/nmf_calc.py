import random
import math
import numpy as np
import matplotlib.pyplot as plt

from modules.nmf import NMF
from modules.data import setup_mnist, csv_out, setup_face
from modules.visualize import show_base_grid, show_reconstruct_pairs, show_base_weight, show_graph
from modules.array import normalize, search_near_imagepair

m = 60000       # 画像数
# r_list = [10, 49, 100, 439, 784, 1000]
r_list = [10, 49, 100, 773, 1000]
# r_list = [439] # 基底数 決め方:(n + m)r < nm
iteration = 200

def main():
    V, labels = setup_mnist(image_num=m)
    # V = setup_face(image_num=m)

    print("V shape:" + str(V.shape))

    for r in r_list:
        print("r = " + str(r))
        nmf = NMF(seed=0)
        nmf.calc(V, r, iteration, save=False, use_cache=False)
        W = nmf.W
        H = nmf.H
        F_list = nmf.loss_LOG
        csv_out('nmf_r' + str(r) + '_m60000.csv', ('iteration', 'F'), (range(1, iteration + 1), F_list))

        # show_graph(range(iteration), F_list, 'iteration', 'F')

        reconstruct_V = np.dot(W, H)
        # show_base_grid(W, r, horizontal_num=5, vertical_num=2, img_cmap="Greens", img_normalize=True, save_img=False, filename=None, show_img=True)
        show_base_grid(W, r, img_cmap="Greens", img_normalize=True, save_img=True, filename='nmf_r' + str(r) + '_grid_m60000.pdf', show_img=False)
        show_reconstruct_pairs(V, reconstruct_V, m, img_cmap='Greys', separate=True, save_img=True, filename='nmf_r' + str(r) + '_reconstruct_m60000.pdf', show_img=False)
        # show_base_weight(V, reconstruct_V, W, H, r, m)

        # TODO: 以下，関数化


if __name__ == "__main__":
    main()
    
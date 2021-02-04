import random
import math
import numpy as np
import matplotlib.pyplot as plt

from modules.nmf import NMF
from modules.data import setup_mnist, csv_out, setup_face
from modules.visualize import show_base_grid, show_reconstruct_pairs, show_base_weight, show_graph

m = 1000       # 画像数
# r_list = [10, 49, 100, 439, 784, 1000]
r_list = [10] # 基底数 決め方:(n + m)r < nm
iteration = 200

def main():
    # V, labels = setup_mnist(image_num=m)
    V = setup_face(image_num=m)

    print("V shape:" + str(V.shape))

    for r in r_list:
        print("r = " + str(r))
        nmf = NMF()
        nmf.calc(V, r, iteration)
        W = nmf.W
        H = nmf.H
        F_list = nmf.loss_LOG
        csv_out('nmf_face_r' + str(r) + '.csv', ('iteration', 'F'), (range(1, iteration + 1), F_list))

        # show_graph(range(iteration), F_list, 'iteration', 'F')

        reconstruct_V = np.dot(W, H)
        show_base_grid(W, r, horizontal_num=5, vertical_num=2, img_cmap="Greens", img_normalize=True, save_img=True, filename='nmf_face_r10_grid.pdf', show_img=False)
        # show_base_grid(W, r, img_cmap="Greens", img_normalize=True)
        show_reconstruct_pairs(V, reconstruct_V, m, img_cmap='Greys', separate=True)
        show_base_weight(V, reconstruct_V, W, H, r, m)
    
if __name__ == "__main__":
    main()
    
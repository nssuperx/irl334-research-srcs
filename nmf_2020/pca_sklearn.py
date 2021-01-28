import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from modules.data import setup_mnist
from modules.array import make_baseGridImage, normalization
from modules.visualize import show_base_grid, show_reconstruct_pairs, show_base_weight

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

n = 28 * 28     # 画素数
m = 1000       # 画像数
r = 100

def main():
    V, label = setup_mnist(image_num=m)
    pca = PCA(n_components=r)
    # 基底ベクトルを取得
    W = pca.fit_transform(V)
    # これを，可視化する．棒グラフを作る．
    H = pca.components_
    print("V shape:" + str(V.shape))
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))
    
    reconstruct_V = pca.inverse_transform(W)
    print("reconstruct_V.shape: " + str(reconstruct_V.shape))
    
    show_base_grid(W, r, img_normalize=True)
    show_reconstruct_pairs(V, reconstruct_V, m, sample_num=5)
    # show_base_weight(V, reconstruct_V, W, H, r, m)


if __name__ == "__main__":
    main()
    
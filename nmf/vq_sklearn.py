import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from modules.data import setup_mnist, setup_face
from modules.visualize import show_graph, show_base_grid, show_base_weight

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

# k-means法を使う．
# まだ全然できてない．

n = 28 * 28 # 画素数
m = 10000    # 画像数
r = 49

def main():
    # V, label = setup_mnist(m)
    V = setup_face(m)
    vq = KMeans(n_clusters=r, random_state=0)
    # 基底ベクトルを取得
    W = vq.fit_transform(V)
    H = vq.labels_
    print("V shape:" + str(V.shape))
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))

    show_base_grid(W, r, img_cmap="Greys", img_normalize=True)

if __name__ == "__main__":
    main()
    
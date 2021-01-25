import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

from modules.visualize import show_base_glid, show_reconstruct_pairs, show_base_weight

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

# ValueError: Key backend: 'qtagg' is not a valid value for backend; supported values are
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg',
# 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

n = 28 * 28     # 画素数
m = 10000       # 画像数
r = 100          # 基底数

def main():
    V, V_test, label, label_test = setup_mnist()
    # nmf = NMF(n_components=r, max_iter=10000, beta_loss='frobenius', solver='cd', tol=0.0001, random_state=0)
    nmf = NMF(n_components=r, max_iter=10000, beta_loss='kullback-leibler', solver='mu', tol=0.0001, random_state=0)
    # 基底ベクトルを取得
    W = nmf.fit_transform(V)
    # これを，可視化する．棒グラフを作る．
    H = nmf.components_
    print("V shape:" + str(V.shape))
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))
    print("iter:%d" % (nmf.n_iter_))
    
    reconstruct_V = np.dot(W, H)

    show_base_glid(W, r, img_cmap="Greys")
    show_reconstruct_pairs(V, reconstruct_V, m)
    # show_base_weight(V, reconstruct_V, W, H, r, m)


def setup_mnist():
    """
    sklearnでmnistのデータを作り，numpy配列を作る．
    """
    digits = fetch_openml(name='mnist_784', version=1, data_home="mnist")
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=m, random_state=0, shuffle=True)

    return x_train.T, x_test.T, y_train, y_test

if __name__ == "__main__":
    main()
    
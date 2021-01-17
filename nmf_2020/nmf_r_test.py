import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

n = 28 * 28 # 画素数
m = 10000    # 画像数
r = 50

def main():
    V, V_test, label, label_test = setup_mnist()
    iter_LOG = []
    F_LOG = []
    for i in range(1, r):
        # n_components:基底数, max_iter:最大繰り返し回数, beta_loss:目的関数, tol:収束許容範囲, random_state:シード値
        nmf = NMF(n_components=i, max_iter=10000, beta_loss='frobenius', tol=0.0001, random_state=0)
        W = nmf.fit_transform(V)    # 基底ベクトルを取得
        H = nmf.components_         # 重み

        print("V shape:" + str(V.shape))
        print("W shape:" + str(W.shape))
        print("H shape:" + str(H.shape))
        print("r = %d, iter:%d, error:%f" % (i, nmf.n_iter_, nmf.reconstruction_err_))
        iter_LOG.append(nmf.n_iter_)
        F_LOG.append(nmf.reconstruction_err_)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(range(1, r), iter_LOG)
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(range(1, r), F_LOG)
    plt.show()    

def setup_mnist():
    """
    sklearnでmnistのデータを作り，numpy配列を作る．
    """
    digits = fetch_openml(name='mnist_784', version=1, data_home="mnist")
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=m, random_state=0, shuffle=True)

    return x_train.T, x_test.T, y_train, y_test

if __name__ == "__main__":
    main()
    
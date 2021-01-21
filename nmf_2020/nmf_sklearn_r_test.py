import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

from data_function import setup_mnist, csv_make_labels, csv_out_row

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

n = 28 * 28     # 画素数
m = 10000       # 画像数
r = 100         # 基底数

def main():
    V, label = setup_mnist(image_num=m)
    iter_LOG = []
    F_LOG = []
    out_filename = 'nmf_sklearn_r_test.csv'
    csv_make_labels(out_filename, ('r', 'iter', 'F'))     # csvのラベルをつける
    for i in range(1, r+1):
        # n_components:基底数, max_iter:最大繰り返し回数, beta_loss:目的関数, solver:ソルバー, tol:収束許容範囲, random_state:シード値
        nmf = NMF(n_components=i, max_iter=10000, beta_loss='frobenius', solver='cd', tol=0.0001, random_state=0)
        # nmf = NMF(n_components=i, max_iter=10000, beta_loss='frobenius', solver='mu', tol=0.0001, random_state=0)
        # nmf = NMF(n_components=i, max_iter=10000, beta_loss='kullback-leibler', solver='mu', tol=0.0001, random_state=0)
        W = nmf.fit_transform(V)    # 基底ベクトルを取得
        H = nmf.components_         # 重み

        print("V shape:" + str(V.shape))
        print("W shape:" + str(W.shape))
        print("H shape:" + str(H.shape))
        print("r = %d, iter:%d, error:%f" % (i, nmf.n_iter_, nmf.reconstruction_err_))
        iter_LOG.append(nmf.n_iter_)
        F_LOG.append(nmf.reconstruction_err_)

        # csv書き出し．処理に相当な時間がかかるのでバックアップをとっておく．
        csv_out_row(out_filename, (i, nmf.n_iter_, nmf.reconstruction_err_))

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(range(1, r+1), iter_LOG)
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(range(1, r+1), F_LOG)
    plt.show()

if __name__ == "__main__":
    main()
    
import numpy as np
import matplotlib.pyplot as plt

from modules.nmf import NMF
from modules.data_function import setup_mnist, csv_make_labels, csv_out_row

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

m = 1000       # 画像数
r = 10         # 基底数

iteration = 500

def main():
    V, label = setup_mnist(image_num=m)
    print("V shape:" + str(V.shape))

    F_LOG = []
    out_filename = 'nmf_r_test_iter' + str(iteration) + '.csv'
    csv_make_labels(out_filename, ('r', 'F'))     # csvのラベルをつける

    for i in range(1, r+1):
        nmf = NMF()
        nmf.calc(V, r=i, iteration=iteration)
        print("W shape:" + str(nmf.W.shape))
        print("H shape:" + str(nmf.H.shape))
        print("r = %d, error:%f" % (i, nmf.loss_LOG[-1]))
        F_LOG.append(nmf.loss_LOG[-1])

        # csv書き出し．処理に相当な時間がかかるのでバックアップをとっておく．
        csv_out_row(out_filename, (i, nmf.loss_LOG[-1]))

    plt.plot(range(1, r+1), F_LOG)
    plt.show()

if __name__ == "__main__":
    main()
    
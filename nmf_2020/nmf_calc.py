import random
import math
import numpy as np
import matplotlib.pyplot as plt

from nmf import NMF
from data_function import setup_mnist

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

m = 1000       # 画像数
r = 10         # 基底数
iteration = 50

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

    # 確認!!
    sample_index = np.random.randint(0,m)
    print(labels[sample_index])
    fig = plt.figure()
    original_npimg = V[:,sample_index].reshape((28, 28))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(original_npimg, cmap="Greys_r")
    
    restore_npimg = np.dot(W,H[:,sample_index]).reshape((28, 28))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(restore_npimg, cmap="Greys_r")
    plt.show()

    # 基底画像の確認
    show_W_img_num = 10
    fig = plt.figure()
    for i in range(show_W_img_num):
        W_img = W[:,i].reshape((28, 28))
        ax = fig.add_subplot(2,5,i+1)
        ax.imshow(W_img, cmap="Greys_r")
    plt.show()

if __name__ == "__main__":
    main()
    
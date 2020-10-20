import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

n = 28 * 28 # 画素数
m = 10000    # 画像数
r = 10

def main():
    V, V_test, label, label_test = setup_mnist()
    nmf = NMF(n_components=r, max_iter=400)
    # nmf.fit(V)
    print("V shape:" + str(V.shape))

    W = nmf.fit_transform(V)
    print("W shape:" + str(W.shape))

    # 基底ベクトルを取得
    H = nmf.components_
    print("H shape:" + str(H.shape))

    print("iter:%d" % (nmf.n_iter_))
    
    H_image = H.reshape(r, 28, 28)

    # 画像の表示
    fig = plt.figure()
    for i in range(0, r):
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(H_image[i])
    plt.show()
    
    # このW_testのヒストグラムを書いてみる．
    W_test = nmf.transform(V_test)
    print(W_test[0])
    print(W_test[1])
    print(W_test[2])
    print(W_test[3])


def setup_mnist():
    """
    sklearnでmnistのデータを作り，numpy配列を作る．
    """
    digits = fetch_openml(name='mnist_784', version=1, data_home="mnist")
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=m, random_state=0, shuffle=True)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    main()
    
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
m = 1000    # 画像数
r = 10

def main():
    V, labels, _, _ = setup_mnist()
    nmf = NMF(n_components=r)
    nmf.fit(V)

    print("--- matrix W ---")
    W = nmf.fit_transform(V)
    print(W)

    print("--- matrix H ---")
    H = nmf.components_
    print(H)

    print("--- matrix WH ---")
    WH = np.dot(W, H)
    print(WH[0])

    H_image = H.reshape(r, 28, 28)

    # 画像の表示
    fig = plt.figure()
    for i in range(0, r):
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(H_image[i])

    plt.show()
    


def setup_mnist():
    """
    sklearnでmnistのデータを作り，numpy配列を作る．
    
    Returns:
        mnist_image:
            m * n次元のmnistのnumpy配列
        labels:
            ラベルのnumpy配列
    """
    digits = fetch_openml(name='mnist_784', version=1)
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, max_iter=1000, random_state=0)

    # この辺で扱いやすいようデータを絞るよう書く．

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    main()
    
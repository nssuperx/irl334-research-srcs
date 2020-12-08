import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

# EPSILON = np.finfo(np.float32).eps
epsilon = 1e-7

n = 28 * 28 # 画素数
m = 1000    # 画像数
r = 10

def main():
    mnist_image, labels = setup_mnist()
    V = mnist_image.T
    # print(V[0])
    W = np.random.rand(n, r)
    H = np.random.rand(r, m)

    print("V shape:" + str(V.shape))
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))
    
    for i in range(10):
        print("iter:%d" % (i))
        W, H = update(V, W, H)

    """
    確認!!
    tmp = np.random.randint(0,m)
    npimg = V[tmp].reshape((28, 28))
    print(labels[tmp])
    plt.imshow(npimg, cmap='gray')
    plt.show()
    """


def setup_mnist():
    """
    pytorchを使ってmnistのデータを作り，numpy配列を作る．
    Returns:
        mnist_image:
            m * n次元のmnistのnumpy配列
        labels:
            ラベルのnumpy配列
    """
    mnist_data = MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(mnist_data, batch_size=m, shuffle=False)
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    mnist_image = np.zeros((m, n))
    for i in range(m):
        mnist_image[i] = images[i][0].numpy().reshape(n)

    return mnist_image, labels.numpy()

'''
\mu番目の「例題」 $\mu = 1,2, ..., m$
i番目の「ピクセル」 $i = 1,2, ..., n$
a番目の「基底」 $a = 1,2, ..., r$
'''
# 値更新
def update(V, W, H):
    print(W)
    input()
    WH = np.dot(W, H) + epsilon
    print("WH shape:" + str(WH.shape))
    for i in range(n):
        for a in range(r):
            tmp_sum = np.sum((V[i] / WH[i]) * H[a])
            # for mu in range(n):
            #     tmp_sum += (V[i][mu] / WH[i][mu]) * H[a][mu]
            W[i][a] = W[i][a] * tmp_sum

    for i in range(n):
        W_col_sum = np.sum(W[i]) + epsilon
        # for a in range(r):
        #     tmp_sum = 0
        #     for j in range(m):
        #         tmp_sum += W[j][a]
        #     W[i][a] = W[i][a] / tmp_sum
        W[i] = W[i] / W_col_sum

    WH = np.dot(W, H) + epsilon
    for a in range(r):
        for mu in range(m):
            tmp_sum = np.sum(W[:,a] * (V[:,mu] / WH[:,mu]))
            # for i in range(n):
            #     tmp_sum += W[i][a] * V[i][mu] / WH[i][mu]
            H[a][mu] = H[a][mu] * tmp_sum

    # W = W * np.dot(V, H) / np.dot(np.dot(W, H.T), H)
    # H = H * np.dot(W.T, V).T / np.dot(W.T, np.dot(W, H.T)).T
    return W, H


if __name__ == "__main__":
    main()
    
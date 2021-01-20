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

np.random.seed(0)

n = 28 * 28 # 画素数
m = 1000    # 画像数
r = 10

iteration = 50

def main():
    mnist_image, labels = setup_mnist()
    V = mnist_image.T
    # print(V[0])
    W = np.random.rand(n, r)
    H = np.random.rand(r, m)

    print("V shape:" + str(V.shape))
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))
    
    F_list = []

    for i in range(iteration):
        W, H = update(V, W, H)
        F = kl_divergence(V, W, H)
        F_list.append(F)
        print("iter:%d F:%f" % (i, F))

    plt.plot(range(iteration), F_list)
    plt.show()

    # 確認!!
    sample_index = np.random.randint(0,m)
    print(labels[sample_index])
    fig = plt.figure()
    original_npimg = V[:,sample_index].reshape((28, 28))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(original_npimg)
    
    restore_npimg = np.dot(W,H[:,sample_index]).reshape((28, 28))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(restore_npimg)
    plt.show()

    # 基底画像の確認
    show_W_img_num = 10
    fig = plt.figure()
    for i in range(show_W_img_num):
        W_img = W[:,i].reshape((28, 28))
        ax = fig.add_subplot(2,5,i+1)
        ax.imshow(W_img)
    plt.show()


def setup_mnist():
    """
    pytorchを使ってmnistのデータを作り，numpy配列を作る．
    
    Returns
    --------
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

def kl_divergence(V, W, H):
    WH = np.dot(W, H) + epsilon
    log_WH = np.log(WH)
    F = np.sum(np.multiply(V, log_WH) - WH)
    return F

def frobenius_norm(V, W, H):
    pass

'''
\mu番目の「例題」 $\mu = 1,2, ..., m$
i番目の「ピクセル」 $i = 1,2, ..., n$
a番目の「基底」 $a = 1,2, ..., r$
'''
# 値更新
def update(V, W, H):
    """
    NMF更新
    Lee & Seung アルゴリズム

    Parameters
    ----------
    V: numpy.adarray
        オリジナルのデータ
    W: numpy.adarray
        基底
    H: numpy.adarray
        重み

    Returns
    --------
    W: numpy.adarray
        基底
    H: numpy.adarray
        重み
    """
    WH = np.dot(W, H) + epsilon
    W = W * np.dot(V / WH, H.T)

    W_tmp = np.sum(W, axis=0)
    W = W / np.tile(W_tmp, (n, 1))

    WH = np.dot(W, H) + epsilon
    H = H * np.dot(W.T, (V/WH))
    
    return W, H


if __name__ == "__main__":
    main()
    
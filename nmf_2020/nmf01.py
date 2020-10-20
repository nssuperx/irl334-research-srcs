import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

n = 28 * 28 # 画素数
m = 1000    # 画像数
r = 10

def main():
    V, labels = setup_mnist()
    # V = np.dot(V, 256)
    print(V[0])
    W = np.random.rand(m, r)
    H = np.random.rand(n, r)
    
    for i in range(10):
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

# 値更新
def update(V, W, H):
    print(W)
    print(H)
    input()
    W = W * np.dot(V, H) / np.dot(np.dot(W, H.T), H)
    H = H * np.dot(W.T, V).T / np.dot(W.T, np.dot(W, H.T)).T
    return W, H


if __name__ == "__main__":
    main()
    
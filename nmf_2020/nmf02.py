import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
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

iteration = 50

def main():
    mnist_image, labels = setup_mnist()
    V = torch.tensor(mnist_image.T)   # 正規化済み[0,1]

    n = 4
    m = 3
    r = 2
    V = torch.rand((n,m), requires_grad=True)
    W = torch.rand((n,r), requires_grad=True)
    H = torch.rand((r,m), requires_grad=True)

    print(V)
    print(V.shape)
    print(W)
    print(W.shape)
    print(H)
    print(H.shape)
    print(torch.matmul(W, H))
    print((torch.matmul(W, H)).shape)
    print(torch.mul(V, (torch.matmul(W, H))))
    print((torch.mul(V, (torch.matmul(W, H)))).shape)


    """
    for i in range(iteration):
        # 距離を計算
        distance = torch.dist(V, W * H)
        dist_LOG.append(distance.data)

        # 微分
        distance.backward()

        x_LOG.append(x.data.clone())
        y_LOG.append(y.data.clone())

        # 引く（勾配の向きにずらす）
        x.data.sub_(mu * x.grad.data)
        y.data.sub_(mu * y.grad.data)

        # 微分をゼロに．ここよくわからない．
        x.grad.data.zero_()
        y.grad.data.zero_()

        if((i+1) % 10 == 0):
            print("iter:" + str(i) + "   x:"+ str(x.data) + "   y:"+ str(y.data))
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

if __name__ == "__main__":
    main()
    
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
r = 50

mu = 0.01
iteration = 10000

# setup function
mseLoss = torch.nn.MSELoss()

def main():
    mnist_image, labels = setup_mnist()
    V = mnist_image.T   # 正規化済み[0,1]

    W = torch.rand((n,r), requires_grad=True)
    H = torch.rand((r,m), requires_grad=True)

    print("V.shape: " + str(V.shape))
    print("W.shape: " + str(W.shape))
    print("H.shape: " + str(H.shape))

    F_LOG = []

    for i in range(iteration):
        # 距離を計算
        # F = kl_divergence(V, W, H)
        F = frobenius_norm(V, W, H)
        # F = mse_loss(V, W, H)
        F_LOG.append(F.data)

        # 微分
        F.backward()

        # print(W.T[1])
        # input()
        
        # 引く（勾配の向きにずらす）
        W.data.sub_(mu * W.grad.data)
        H.data.sub_(mu * H.grad.data)

        # 微分をゼロに．ここよくわからない．
        W.grad.zero_()
        H.grad.zero_()
        
        """
        with torch.no_grad(): 
            # 引く（勾配の向きにずらす）
            W.data.sub_(mu * W.grad.data)
            H.data.sub_(mu * H.grad.data)

            # 微分をゼロに．ここよくわからない．
            W.grad.data.zero_()
            H.grad.data.zero_()
        """

    plt.plot(range(iteration), F_LOG)
    plt.show()

    sample_index = random.randrange(0, m)
    print(labels[sample_index])
    fig = plt.figure()
    original_img = V[:,sample_index].reshape((28, 28))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(original_img.data)
    
    restore_img = torch.matmul(W,H[:,sample_index]).reshape((28, 28))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(restore_img.data)
    plt.show()

    # 基底画像の確認
    show_W_img_num = 10
    fig = plt.figure()
    for i in range(show_W_img_num):
        W_img = W[:,i].reshape((28, 28))
        ax = fig.add_subplot(2,5,i+1)
        ax.imshow(W_img.data)
    plt.show()

    print(W.T[0]) # 負の値もあったしスパースじゃない???


def kl_divergence(V, W, H):
    WH = torch.matmul(W, H)
    log_WH = torch.log(WH)
    F = torch.sum(torch.mul(V, log_WH) - WH)
    # F = torch.sum(V * torch.log(V / WH)) - torch.sum(V) + torch.sum(WH)
    return F

def frobenius_norm(V, W, H):
    WH = torch.matmul(W,H)
    F = torch.linalg.norm(V - WH)
    # F = torch.sqrt(torch.sum(torch.sub(V, WH)**2))
    return F

def mse_loss(V, W, H):
    """
    平均二乗誤差
    """
    WH = torch.matmul(W,H)
    # F = torch.mean(torch.sub(V, WH)**2)
    F = mseLoss(V, WH)

    """
    ニ乗和のつもりだったが違った．値が大きくなりすぎてた．
    F = torch.sum(torch.sub(V, WH)**2)
    """

    return F


def setup_mnist():
    """
    mnistのデータを作る．
    Returns:
        mnist_image:
            m * n次元のmnistのtensor型のデータ
        labels:
            ラベル
    """
    mnist_data = MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(mnist_data, batch_size=m, shuffle=False)
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    mnist_image = torch.zeros((m, n))
    for i in range(m):
        mnist_image[i] = images[i][0].reshape(n)

    return mnist_image, labels


if __name__ == "__main__":
    main()
    
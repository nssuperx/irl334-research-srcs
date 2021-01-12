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
iteration = 500

def main():
    mnist_image, labels = setup_mnist()
    V = torch.tensor(mnist_image.T)   # 正規化済み[0,1]

    # n = 4
    # m = 3
    # r = 2
    # V = torch.rand((n,m))
    W = torch.rand((n,r), requires_grad=True)
    H = torch.rand((r,m), requires_grad=True)

    print("V.shape: " + str(V.shape))
    print("W.shape: " + str(W.shape))
    print("H.shape: " + str(H.shape))
    print("(torch.matmul(W, H)).shape: " + str((torch.matmul(W, H)).shape))
    print("(torch.mul(V, (torch.matmul(W, H)))).shape: " + str((torch.mul(V, (torch.matmul(W, H)))).shape))

    F_LOG = []

    for i in range(iteration):
        # 距離を計算
        # KL-divergence
        """
        WH = torch.matmul(W, H) + epsilon
        log_WH = torch.log(WH)
        F = torch.sum(torch.mul(V, log_WH) - WH)
        """

        # Frobenius norm
        WH = torch.matmul(W,H) + epsilon
        # 以下2つ同じ
        # F = torch.linalg.norm(V - WH)
        F = torch.sqrt(torch.sum(torch.sub(V, WH)**2))

        # 2乗和
        # F = torch.sum(torch.sub(V, WH)**2)

        F_LOG.append(F.data)


        # 微分
        F.backward()

        # print(W.grad)
        # print(H.grad)
        # input()

        # 引く（勾配の向きにずらす）
        W.data.sub_(mu * W.grad.data)
        H.data.sub_(mu * H.grad.data)

        # 微分をゼロに．ここよくわからない．
        W.grad.data.zero_()
        H.grad.data.zero_()

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
    
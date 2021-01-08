import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

epsilon = 1e-7

n = 28 * 28 # 画素数
m = 1000    # 画像数
r = 10

class NMF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, W, H):
        ctx.save_for_backward(V, W, H)
        WH = np.dot(W, H) + epsilon
        log_WH = np.log(WH)
        F = np.sum(np.multiply(V, log_WH) - WH)
        return F

    @staticmethod
    def backward(ctx, output):
        V, W, H = ctx.saved_tensors
        dF_dW = None
        dF_dH = None

        return dF_dW, dF_dH

def main():
    mnist_image, labels = setup_mnist()
    V = mnist_image.T
    # print(V[0])
    W = np.random.rand(n, r)
    H = np.random.rand(r, m)

    F = cost_func_forward(V,W,H)
    print(type(F))
    print(F)


# 目的関数はこれ
def cost_func_forward(V, W, H):
    WH = np.dot(W, H) + epsilon
    log_WH = np.log(WH)
    F = np.sum(np.multiply(V, log_WH) - WH)
    return F

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

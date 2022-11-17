import torch
from torch import nn
from self_made_nn import TSigmoid
import matplotlib.pyplot as plt


def main():
    x = torch.arange(-10, 10, 0.02)
    print(x)
    sigmoid = nn.Sigmoid()
    y1 = sigmoid(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(x, y1)
    plt.show()

    tsigmoid = TSigmoid(0.1, 0.5)
    y2 = tsigmoid(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(x, y2)
    plt.show()

    tsigmoid = TSigmoid(2, 0.5)
    y3 = tsigmoid(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(x, y3)
    plt.show()


if __name__ == "__main__":
    main()

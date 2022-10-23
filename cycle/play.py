import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_dataset = datasets.MNIST(
    root='../pt_datasets',
    train=True,
    download=True,
    transform=ToTensor()
)
test_dataset = datasets.MNIST(
    root='../pt_datasets',
    train=False,
    download=True,
    transform=ToTensor()
)

trainloader = DataLoader(train_dataset, shuffle=True)
testloader = DataLoader(test_dataset, shuffle=False)
MNIST_classes = datasets.MNIST.classes

B_classes = 31


class ArgMax(nn.Module):
    """argmaxをnn.Module化したもの
    NOTE: 必要ないかもしれない
    """

    def __init__(self):
        super(ArgMax, self).__init__()

    def forward(self, input: torch.Tensor):
        return input.argmax()


class ClampArg(nn.Module):
    """出力番号（要素）を0, 1の間でクランプする
    0は0，1以上は1にできる
    NOTE: 必要ないかもしれない
    """

    def __init__(self):
        super(ClampArg, self).__init__()

    def forward(self, input: torch.Tensor):
        return input.clamp(min=0, max=1)


class HiddenBrick(nn.Module):
    """隠れ層の役割のBrick
    """
    def __init__(self):
        super(HiddenBrick, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, B_classes)
        # self.maxcell = nn.MaxPool1d(B_classes)
        self.argmax = ArgMax()
        self.clamp = ClampArg()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.maxcell(x)
        x = self.argmax(x)
        x = self.clamp(x)
        return x


class OutBrick(nn.Module):
    def __init__(self):
        super(OutBrick, self).__init__()

    def forward(self):
        pass


def train_loop(dataloader: DataLoader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    model = HiddenBrick()
    print(model)

    learning_rate = 1e-3
    batch_size = 64

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        test_loop(testloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()

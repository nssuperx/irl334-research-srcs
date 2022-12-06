import os
import datetime
import json
from typing import NamedTuple
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from modules.visualize import show_brick_weight_allInOnePicture
import modules.self_made_nn as smnn
import matplotlib.pyplot as plt


class ExperimentInfo(NamedTuple):
    title: str
    description: str


class HyperParameter(NamedTuple):
    seed: int
    kernel_size: int
    stride: int
    MV_classes: int
    MV_bricks: int
    bricks_out: int
    learning_rate: float
    batch_size: int
    epochs: int


ei = ExperimentInfo("PartedMultiValue", "none")
hp = HyperParameter(2022, 8, 4, 15, 10, 5, 1e-2, 1, 100)

torch.manual_seed(hp.seed)

train_dataset = datasets.MNIST(
    root='../pt_datasets',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_dataset = datasets.MNIST(
    root='../pt_datasets',
    train=False,
    download=True,
    transform=ToTensor(),
)

MNIST_classes = datasets.MNIST.classes

# 教師ラベルを整形するときに使う
sp_list = list(range(len(MNIST_classes) + 1))
sp_list[0], sp_list[-1] = sp_list[-1], sp_list[0]
slice_pattern = tuple(sp_list)


class PartedMultiValueNet(nn.Module):
    def __init__(self):
        """部分で見ていく．バッチ処理未対応
        """
        super(PartedMultiValueNet, self).__init__()
        self.mvbrick = smnn.MultiValueBrick(hp.kernel_size * hp.kernel_size, hp.MV_bricks, hp.MV_classes)
        kernel_num = (28 - hp.kernel_size + hp.stride) // hp.stride
        self.fc1 = nn.Linear(kernel_num**2 * hp.MV_bricks, hp.bricks_out)
        self.fc2 = nn.Linear(hp.bricks_out, len(MNIST_classes) + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = x.unfold(2, hp.kernel_size, hp.stride).unfold(3, hp.kernel_size, hp.stride)
        x = x.permute([0, 2, 3, 1, 4, 5]).reshape(-1, 1, hp.kernel_size, hp.kernel_size)
        x = self.mvbrick(x)
        x = x.flatten().reshape(hp.batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def train_loop(dataloader: DataLoader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)

        # 教師ラベルをone-hotにする．
        label = F.one_hot(y, len(MNIST_classes) + 1).to(torch.float32)

        # 0番目は該当なし，最後は数字の0に割り当てられるように入れ替える
        label = label[:, slice_pattern]

        loss = loss_fn(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch * int(dataloader.batch_size) % 10000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            # 教師ラベルを整形する処理
            label = F.one_hot(y, len(MNIST_classes) + 1).to(torch.float32)
            label = label[:, slice_pattern]

            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(dim=-1) == torch.where(y == 0, 10, y)).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def plot_graph(acc: list, loss: list, path: str):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(acc)
    ax2.plot(loss)
    ax1.set_title("Accuracy")
    ax2.set_title("Avg loss")
    fig.tight_layout()
    fig.savefig(f"{path}/acc.pdf")


def experiment_setup():
    workdir = f"./out/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(workdir)

    with open(f"{workdir}/info.json", "w") as f:
        json.dump(ei._asdict() | hp._asdict(), f, indent=4)

    return workdir


def main():
    workdir = experiment_setup()

    trainloader = DataLoader(train_dataset, hp.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, hp.batch_size, shuffle=False)
    model = PartedMultiValueNet()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate)

    accuracy = []
    avg_loss = []

    # 何もしていない最初の状態
    show_brick_weight_allInOnePicture(model.mvbrick.fc, hp.MV_bricks, hp.MV_classes,
                                      hp.kernel_size, hp.kernel_size, 0, workdir)

    for t in range(hp.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        acc, al = test_loop(testloader, model, loss_fn)
        accuracy.append(acc)
        avg_loss.append(al)
        show_brick_weight_allInOnePicture(model.mvbrick.fc, hp.MV_bricks, hp.MV_classes,
                                          hp.kernel_size, hp.kernel_size, t + 1, workdir)
    print("Done!")

    plot_graph(accuracy, avg_loss, workdir)


if __name__ == "__main__":
    main()

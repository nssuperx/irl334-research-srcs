import os
import datetime
import json
from typing import NamedTuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import modules.self_made_nn as smnn
from modules.cycle_toy_datasets import VerticalLine, XORProblem
import matplotlib.pyplot as plt


class ExperimentInfo(NamedTuple):
    title: str
    description: str


class HyperParameter(NamedTuple):
    seed: int
    B_classes: int
    B_bricks: int
    linear_hidden: int
    learning_rate: float
    batch_size: int
    epochs: int


ei = ExperimentInfo("MultiValue", "Cycle test")
hp = HyperParameter(2022, 3, 3, 20, 1e-2, 1, 100000)

torch.manual_seed(hp.seed)

toy_datasets = VerticalLine()
xor_datasets = XORProblem()


class MultiValueNet(nn.Module):
    def __init__(self):
        super(MultiValueNet, self).__init__()
        self.mvbrick1 = smnn.MultiValueBrick(3, 1, hp.B_classes)
        self.mvbrick2 = smnn.MultiValueBrick(3, 1, hp.B_classes)
        self.mvbrick3 = smnn.MultiValueBrick(3, 1, hp.B_classes)
        self.out = smnn.OutBrick(hp.B_bricks, 2)

    def forward(self, x: torch.Tensor):
        x1 = self.mvbrick1(x[:, 0, :])
        x2 = self.mvbrick2(x[:, 1, :])
        x3 = self.mvbrick3(x[:, 2, :])
        x = torch.cat((x1, x2, x3)).reshape(hp.batch_size, 3)
        x = self.out(x)
        return x


class TestFullyConnectNet(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(TestFullyConnectNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hp.linear_hidden)
        self.hidden_softmax = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(hp.linear_hidden, out_features)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x.to(dtype=torch.float32))
        x = self.fc1(x)
        x = self.hidden_softmax(x)
        x = self.fc2(x)
        return x


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
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=-1) == y).type(torch.float32).sum().item()

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

    trainloader = DataLoader(toy_datasets, hp.batch_size, shuffle=True)
    testloader = DataLoader(toy_datasets, hp.batch_size, shuffle=False)
    model = MultiValueNet()
    # model = TestFullyConnectNet(9, 2)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate)

    accuracy = []
    avg_loss = []

    for t in range(hp.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        acc, al = test_loop(testloader, model, loss_fn)
        accuracy.append(acc)
        avg_loss.append(al)
    print("Done!")

    plot_graph(accuracy, avg_loss, workdir)


if __name__ == "__main__":
    main()

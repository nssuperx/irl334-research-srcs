import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

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

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.maxcell(x)
        x = self.argmax(x)
        x = self.clamp(x)
        return x


class OutBrick(nn.Module):
    def __init__(self, in_features: int):
        super(OutBrick, self).__init__()
        self.fc = nn.Linear(in_features, len(MNIST_classes) + 1)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class CycleNet(nn.Module):
    def __init__(self):
        super(CycleNet, self).__init__()
        self.hidden_bricks = nn.ModuleList([HiddenBrick() for i in range(B_classes)])
        self.out = OutBrick(B_classes)

    def forward(self, x: torch.Tensor):
        b_out = torch.empty(B_classes)
        for i, hidden in enumerate(self.hidden_bricks):
            b_out[i] = hidden(x)
        x = self.out(b_out)
        return x


def train_loop(dataloader: DataLoader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)

        # 教師ラベルをone-hotにして0番目と最後を入れ替える
        # 0番目は該当なし，最後は数字の0に割り当てられる
        # TODO: バッチ処理未対応
        label = F.one_hot(y, len(MNIST_classes) + 1)[0].to(torch.float32)  # one-hotにする
        label[0], label[len(label) - 1] = label[len(label) - 1], label[0]
        
        loss = loss_fn(pred, label)

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
    model = CycleNet()
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

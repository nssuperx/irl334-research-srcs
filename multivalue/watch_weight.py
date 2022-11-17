import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from modules.visualize import show_brick_weight_all

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

B_classes = 31
B_Bricks = 20


class ArgMax(nn.Module):
    """argmaxをnn.Module化したもの
    NOTE: 必要ないかもしれない
    """

    def __init__(self):
        super(ArgMax, self).__init__()

    def forward(self, input: torch.Tensor):
        return input.argmax(dim=-1)


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

    def __init__(self, out_features: int):
        super(HiddenBrick, self).__init__()
        self.out_features = out_features
        self.flatten = nn.Flatten()
        # NOTE: メモ参照．HiddenBrickのnn.Linearについて
        self.fc1 = nn.Linear(28 * 28, B_classes * out_features)
        self.argmax = ArgMax()
        self.clamp = ClampArg()

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = x.reshape(x.shape[0], self.out_features, B_classes)
        x = self.argmax(x)
        x = self.clamp(x)
        return x


class OutBrick(nn.Module):
    def __init__(self, in_features: int):
        super(OutBrick, self).__init__()
        self.fc = nn.Linear(in_features, len(MNIST_classes) + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.fc(x.to(dtype=torch.float32))
        x = self.softmax(x)
        return x


class CycleNet(nn.Module):
    def __init__(self):
        super(CycleNet, self).__init__()
        self.hidden_brick = HiddenBrick(B_Bricks)
        self.out = OutBrick(B_Bricks)

    def forward(self, x: torch.Tensor):
        x = self.hidden_brick(x)
        x = self.out(x)
        return x


def train_loop(dataloader: DataLoader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)

        # 教師ラベルをone-hotにする．0番目は該当なし，最後は数字の0に割り当てられる
        label = F.one_hot(y, len(MNIST_classes) + 1).to(torch.float32)

        slice_pattern = list(range(len(MNIST_classes) + 1))
        slice_pattern[0], slice_pattern[-1] = slice_pattern[-1], slice_pattern[0]

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
            slice_pattern = list(range(len(MNIST_classes) + 1))
            slice_pattern[0], slice_pattern[-1] = slice_pattern[-1], slice_pattern[0]
            label = label[:, slice_pattern]

            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(dim=-1) == torch.where(y == 0, 10, y)).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    learning_rate = 1e-3
    batch_size = 10

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = CycleNet()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        test_loop(testloader, model, loss_fn)
        show_weight_cycle_hidden(model.hidden_brick.fc1, B_Bricks, B_classes, t)
    print("Done!")


if __name__ == "__main__":
    main()

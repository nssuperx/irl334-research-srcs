import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

trainset = datasets.MNIST(
    root='../pt_datasets',
    train=True,
    download=True,
    transform=ToTensor()
)
trainloader = DataLoader(trainset, shuffle=True)
testset = datasets.MNIST(
    root='../pt_datasets',
    train=False,
    download=True,
    transform=ToTensor()
)
testloader = DataLoader(testset, shuffle=False)

classes = datasets.MNIST.classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, len(classes))

    def forward(self, x):
        x = torch.flatten(x)
        x = self.fc1(x)
        # x = torch.flatten(x, 1)
        # x = torch.sigmoid(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        return x


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer):
    # writer = SummaryWriter()
    running_loss = 0.0
    for batch, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # MNIST のとき torch.Size([1, 1, 28, 28])
        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch % 2000 == 1999:    # print every 2000 mini-batches
            loss, current = loss.item(), batch * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

        # writer.add_graph(net, inputs)
        # writer.close()


def test(dataloader: DataLoader, model: nn.Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    net = Net()
    learning_rate = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    epochs = 10

    for t in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, net, loss_fn, optimizer)
        test(testloader, net, loss_fn)

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()

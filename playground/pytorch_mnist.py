import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True)
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, shuffle=False)

classes = torchvision.datasets.MNIST.classes

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, len(classes))

    def forward(self, x):
        x = torch.flatten(x)
        # x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        return x

def main():
    net = Net()

    criterion = nn.MSELoss()

    for epoch in range(1):  # loop over the dataset multiple times
        writer = SummaryWriter()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # MNIST のとき torch.Size([1, 1, 28, 28])
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            writer.add_graph(net, inputs)
            writer.close()

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

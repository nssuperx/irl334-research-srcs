from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

trainset = datasets.MNIST(
    root='../pt_datasets',
    train=True,
    download=True,
    transform=ToTensor()
)

testset = datasets.MNIST(
    root='../pt_datasets',
    train=False,
    download=True,
    transform=ToTensor()
)


def test_flatten():
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)

    tl_iter = iter(trainloader)
    data = next(tl_iter)

    print(f"not flatten: {data[0].shape}")

    print(f"flatten: {data[0].flatten(start_dim=1).shape}")

    flatten = nn.Flatten()  # default start_dim=1
    print(f"flatten: {flatten(data[0]).shape}")


if __name__ == "__main__":
    test_flatten()

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time

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


def test_mnist():
    batch_size = 10
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    class_count = [0 for i in range(10)]
    for batch, data in enumerate(trainloader):
        if batch * batch_size >= 1000:
            break

        n = data[1]

        for i in n:
            class_count[i] += 1

    print(class_count)
    print(sum(class_count))


def calc_time_batchsize():
    batch_size = 1
    trainloader_batchsize1 = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    # count = 0
    start = time.perf_counter()
    for batch, data in enumerate(trainloader_batchsize1):
        pass
        # count += len(data[0])
    print(f"batch size {batch_size}: {time.perf_counter() - start}")

    batch_size = 10
    trainloader_batchsize10 = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    # count = 0
    start = time.perf_counter()
    for batch, data in enumerate(trainloader_batchsize10):
        pass
        # count += len(data[0])
    print(f"batch size {batch_size}: {time.perf_counter() - start}")

    batch_size = 64
    trainloader_batchsize64 = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    # count = 0
    start = time.perf_counter()
    for batch, data in enumerate(trainloader_batchsize64):
        pass
        # count += len(data[0])
    print(f"batch size {batch_size}: {time.perf_counter() - start}")

    batch_size = 100
    trainloader_batchsize100 = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    # count = 0
    start = time.perf_counter()
    for batch, data in enumerate(trainloader_batchsize100):
        pass
        # count += len(data[0])
    print(f"batch size {batch_size}: {time.perf_counter() - start}")


if __name__ == "__main__":
    # test_mnist()
    calc_time_batchsize()

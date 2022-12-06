import torch
import torchvision
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

MNIST_classes = datasets.MNIST.classes

batch_size = 1

dl = DataLoader(train_dataset, batch_size, shuffle=False)

it = iter(dl)
image, _ = next(it)

torchvision.utils.save_image(image, "orig.png", normalize=True)

miniimg = image.unfold(2, 4, 2).unfold(3, 4, 2)  # torch.Size([1, 1, 13, 13, 4, 4])

miniimg = miniimg.permute([0, 2, 3, 1, 4, 5])
miniimg = miniimg.reshape(-1, 1, 4, 4)
torchvision.utils.save_image(miniimg, "test.png", nrow=13, normalize=True)

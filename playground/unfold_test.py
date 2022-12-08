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

batch_size = 10

dl = DataLoader(train_dataset, batch_size, shuffle=False)

it = iter(dl)
image, _ = next(it)

torchvision.utils.save_image(image, "orig.png", normalize=True)

miniimg: torch.Tensor = image.unfold(2, 4, 2).unfold(3, 4, 2)  # torch.Size([1, 1, 13, 13, 4, 4])

# 直感でわかりやすいUnfoldした画像の作り方．オリジナル画像1枚を単位としてまとめたもの
miniimg = miniimg.permute((0, 2, 3, 1, 4, 5))
miniimg = miniimg.reshape(batch_size, 13 * 13, 4, 4)
torchvision.utils.save_image(miniimg[1].unsqueeze(dim=1), "test.png", nrow=13, normalize=True)
test = miniimg.detach().clone()[1]

# バッチ枚数分画像を重ねて，重ねたままカットして，それを区画ごとにまとめるイメージ
miniimg = miniimg.permute((1, 0, 2, 3))
torchvision.utils.save_image(miniimg[:, 1].unsqueeze(dim=1), "test2.png", nrow=13, normalize=True)
test2 = miniimg.detach().clone()[:, 1]

diff = test - test2
print(diff.sum())
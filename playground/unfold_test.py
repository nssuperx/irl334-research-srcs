import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image

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
kernel_size = 8
stride = 4
kernel_row = (28 - kernel_size + stride) // stride

dl = DataLoader(train_dataset, batch_size, shuffle=False)

it = iter(dl)
image, _ = next(it)

torchvision.utils.save_image(image, f"orig{batch_size}.png", normalize=True)

miniimg: torch.Tensor = image.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)

# 直感でわかりやすいUnfoldした画像の作り方．オリジナル画像1枚を単位としてまとめたもの
miniimg = miniimg.permute((0, 2, 3, 1, 4, 5))
miniimg = miniimg.reshape(batch_size, kernel_row**2, kernel_size, kernel_size)

for i, mi in enumerate(miniimg[0]):
    img = Image.fromarray(mi.numpy() * 255).convert("L")
    img.save(f"unfold-{i}.png")

torchvision.utils.save_image(miniimg[0].unsqueeze(dim=1), "test.png", nrow=kernel_row, normalize=True)
test = miniimg.detach().clone()[1]

# バッチ枚数分画像を重ねて，重ねたままカットして，それを区画ごとにまとめるイメージ
miniimg = miniimg.permute((1, 0, 2, 3))
torchvision.utils.save_image(miniimg[:, 1].unsqueeze(dim=1), "test2.png", nrow=kernel_row, normalize=True)
test2 = miniimg.detach().clone()[:, 1]

diff = test - test2
print(diff.sum())

import torch
from torch import nn
import torchvision


def show_brick_weight(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int, target: int = 1):
    """
    MultiValueNetの結合の重みを見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))
    target_weight = weight[target]
    img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
    torchvision.utils.save_image(img[0], f"./out/w{target:02}epoch{epoch_times:06}.png")


def show_brick_weight_all(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int):
    """
    MultiValueNetの結合の重み全部見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))

    for target, target_weight in enumerate(weight):
        img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
        torchvision.utils.save_image(img[0], f"./out/w{target:02}epoch{epoch_times:06}.png")


def show_brick_weight_allInOnePicture(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int):
    """
    MultiValueNetの結合の重み全部を1枚の画像にして出力．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))
    imgs = []

    for target, target_weight in enumerate(weight):
        img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
        imgs.append(img[0])

    grid_imgs = torchvision.utils.make_grid(torch.stack(imgs).unsqueeze(dim=1), nrow=4)
    torchvision.utils.save_image(grid_imgs, f"./out/epoch{epoch_times:06}.png")

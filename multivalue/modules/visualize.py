import math
import torch
from torch import nn
import torchvision


def show_brick_weight(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int, target: int, path: str):
    """
    MultiValueNetの結合の重みを見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))
    target_weight = weight[target]
    img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
    torchvision.utils.save_image(img[0], f"{path}/w{target:02}epoch{epoch_times:06}.png")


def show_brick_weight_all(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int, path: str):
    """
    MultiValueNetの結合の重み全部見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))

    for target, target_weight in enumerate(weight):
        img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
        torchvision.utils.save_image(img[0], f"{path}/w{target:02}epoch{epoch_times:06}.png")


def show_brick_weight_allInOnePicture(layer: nn.Linear, out_bricks: int, classes: int,
                                      height: int, width: int, epoch_times: int, path: str):
    """
    MultiValueNetの結合の重み全部を1枚の画像にして出力．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, height, width))
    imgs = []

    for target_weight in weight:
        img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
        imgs.append(img)

    grid_imgs = torchvision.utils.make_grid(torch.stack(imgs), nrow=4)
    torchvision.utils.save_image(grid_imgs, f"{path}/epoch{epoch_times:06}.png")


def show_parted_brick_weight_allInOnePicture(ml: nn.ModuleList, classes: int,
                                             height: int, width: int, epoch_times: int, path: str):
    """
    部分で見るMultiValueNetの結合の重み全部を1枚の画像にして出力．MNIST用．
    moduleListを受け取る
    Args:
        ml (nn.ModuleList)
    """
    imgs = []

    for layer in ml:
        weight = layer.fc.weight.detach().clone()
        weight = weight.reshape(classes, height, width)
        img = torchvision.utils.make_grid(weight.unsqueeze(dim=1), normalize=True,
                                          nrow=int(math.ceil(math.sqrt(classes))))
        imgs.append(img)

    grid_imgs = torchvision.utils.make_grid(torch.stack(imgs), nrow=int(math.sqrt(len(ml))))
    torchvision.utils.save_image(grid_imgs, f"{path}/epoch{epoch_times:06}.png")

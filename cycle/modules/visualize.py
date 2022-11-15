from torch import nn
import torchvision


def show_weight_cycle_hidden(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int):
    """
    cyclenetの結合の重みを見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    target = 6
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))
    target_weight = weight[target]
    img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
    torchvision.utils.save_image(img[0], f"./out/w{target:02}epoch{epoch_times:06}.png")


def show_weight_all_cycle_hidden(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int):
    """
    cyclenetの結合の重み全部見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat = layer.weight.detach().clone()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))

    for target, target_weight in enumerate(weight):
        img = torchvision.utils.make_grid(target_weight.unsqueeze(dim=1), normalize=True)
        torchvision.utils.save_image(img[0], f"./out1/w{target:02}epoch{epoch_times:06}.png")
    return

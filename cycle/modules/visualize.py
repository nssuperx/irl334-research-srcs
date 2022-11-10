from torch import nn
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def show_weight_cycle_hidden(layer: nn.Linear, classes: int, epoch_times: int):
    """
    cyclenetの結合の重みを見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    target = 3
    weight_flat: NDArray = layer.weight.detach().numpy().copy()
    weight = weight_flat.reshape((classes, 28, 28))
    target_weight = weight[target]
    row, col = 4, 10

    fig = plt.figure()
    for i, img in enumerate(target_weight):
        ax = fig.add_subplot(row, col, i + 1)
        ax.set_axis_off()
        ax.imshow(img, cmap="gray", interpolation="none")
    plt.savefig(f"./out/w{target:02}epoch{epoch_times:06}.png")
    return


def show_weight_all_cycle_hidden(layer: nn.Linear, out_bricks: int, classes: int, epoch_times: int):
    """
    cyclenetの結合の重み（全部）を見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat: NDArray = layer.weight.detach().numpy().copy()
    weight = weight_flat.reshape((out_bricks, classes, 28, 28))
    row, col = 4, 10

    for num, target_weight in enumerate(weight):
        fig = plt.figure()
        for i, img in enumerate(target_weight):
            ax = fig.add_subplot(row, col, i + 1)
            ax.set_axis_off()
            ax.imshow(img, cmap="gray", interpolation="none")
        plt.savefig(f"./out/w{num:02}epoch{epoch_times:06}.png")
    return

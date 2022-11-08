from torch import nn
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def show_network_weight(layer: nn.Linear):
    """
    結合の重みを見る．MNIST用．
    Args:
        layer (nn.Linear)
    """
    weight_flat: NDArray = layer.weight.detach().numpy().copy()
    weight = weight_flat.reshape((weight_flat.shape[0], 28, 28))
    row, col = 4, 10

    fig = plt.figure()
    for i, img in enumerate(weight):
        ax = fig.add_subplot(row, col, i + 1)
        ax.set_axis_off()
        ax.imshow(img, cmap="gray", interpolation="none")
    plt.show()
    return

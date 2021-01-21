import numpy as np
import csv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

def setup_mnist(image_num):
    """
    pytorchを使ってmnistのデータを作り，numpy配列を作る．

    Parameters
    ----------
    image_num : int
        使用する画像の枚数

    
    Returns
    --------
    mnist_image:
        n * m次元のmnistのnumpy配列
    labels:
        ラベルのnumpy配列
    """
    mnist_data = MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(mnist_data, batch_size=image_num, shuffle=False)
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    mnist_image = images.reshape(image_num, images.shape[2] * images.shape[3])
    return mnist_image.numpy().T, labels.numpy()


def csv_make_labels(filename, labels):
    """
    csvファイルの1行目にラベルを書く

    Parameters
    ----------
    filename : str
        出力ファイル名
    labels : tuple (str)
        ラベルのタプル
        例: ('r', 'iter', 'F')
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(labels)


def csv_out_row(filename, out_row):
    """
    1イテレートの結果をcsv形式で追記する．

    Parameters
    ----------
    filename : str
        出力ファイル名
    out_row : tuple
        結果のタプル
        例: (r, iteration, F)
    """
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(out_row)
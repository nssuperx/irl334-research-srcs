import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .array import make_baseGridImage


def show_base_grid(W, r, horizontal_num=None, vertical_num=None, img_normalize=False, img_cmap="PiYG", grid_color="black"):
    """
    基底画像をグリッド状に表示する．

    Parameters
    ----------
    W: numpy.ndarray
        基底画像
    r: int
        基底数
    img_cmap: str
        表示する画像のカラーマップ
    grid_color: str
        グリッドの色
    """
    img_width = int(np.sqrt(W.shape[0]))
    img_height = int(np.sqrt(W.shape[0]))
    if horizontal_num is None or vertical_num is None:
        h_num = int(np.sqrt(r))
        v_num = int(np.sqrt(r))
    else:
        h_num = horizontal_num
        v_num = vertical_num

    # 基底ベクトルを画像のように整形
    W_img = W.T.reshape(r, img_height, img_width)

    # 基底画像をグリッド状に表示
    W_imgs = make_baseGridImage(W_img, h_num, v_num, normalize=img_normalize)
    plt.figure(figsize=(6,6))
    plt.imshow(W_imgs, cmap=img_cmap, extent=(0, img_width * h_num, 0, img_height * v_num))
    plt.xticks(range(0, img_width * h_num, img_width))
    plt.yticks(range(0, img_height * v_num, img_height))
    plt.grid(which="major", color=grid_color, alpha=1.0, linestyle="--", linewidth=1)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.show()


def show_reconstruct_pairs(V, reconstruct_V, m, sample_num=5, img_cmap="Greys", random_select=False, img_height=None, img_width=None):
    """
    オリジナル画像と再構成画像のペアを表示する．

    Parameters
    ----------
    V: numpy.ndarray
        オリジナル画像配列
    reconstruct_V: numpy.ndarray
        再構成画像配列
    m: int
        画像総数
    sumple_num: int
        表示したいペア数
    img_cmap: str
        表示する画像のカラーマップ
    random_select: boolean
        表示する画像をランダムで選ぶかどうか
    img_height, img_width, int
        表示する画像の縦と横のピクセル数
        設定しなかったらsqrt(V.shape[0])
    """
    if random_select:
        sample_index_list = np.random.randint(0, m, size=sample_num)
    else:
        sample_index_list = np.arange(sample_num, dtype='int32')

    if img_height is None or img_width is None:
        height = int(np.sqrt(V.shape[0]))
        width = int(np.sqrt(V.shape[0]))
    else:
        height = int(img_height)
        width = int(img_width)
    
    fig = plt.figure()
    for i in range(0, sample_num):
        img = V[:,sample_index_list[i]].reshape(height, width)
        ax = fig.add_subplot(2,5,i+1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        aximg = ax.imshow(img, cmap=img_cmap)
        # aximg = ax.imshow(img, cmap=img_cmap, vmin=-1, vmax=1)
        # fig.colorbar(aximg, ax=ax)
        img = reconstruct_V[:,sample_index_list[i]].reshape(height, width)
        ax = fig.add_subplot(2,5,i+6)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        aximg = ax.imshow(img, cmap=img_cmap)
        # aximg = ax.imshow(img, cmap=img_cmap, vmin=-1, vmax=1)
        # fig.colorbar(aximg, ax=ax)
    plt.show()


def show_base_weight(V, reconstruct_V, W, H, r, m, sample_num=5, img_cmap="Greys"):
    """
    再構成画像の基底画像と重みの関係を棒グラフで表示する．
    基底数が多くなると，見にくい．

    Parameters
    ----------
    V: numpy.ndarray
        オリジナル画像配列
    reconstruct_V: numpy.ndarray
        再構成画像配列
    W: numpy.ndarray
        基底画像
    H: numpy.ndarray
        重み配列
    r: int
        基底数
    m: int
        画像総数
    sumple_num: int
        表示したいペア数
    img_cmap: str
        表示する画像のカラーマップ
    """

    img_width = int(np.sqrt(W.shape[0]))
    img_height = int(np.sqrt(W.shape[0]))
    W_img = W.T.reshape(r, img_height, img_width)

    sample_index_list = np.random.randint(0, m, size=sample_num)
    figrow_master = sample_num + 1
    figcol_master = r + 2
    fig = plt.figure()
    plt.rcParams['axes.xmargin'] = 0
    gs_master = GridSpec(nrows=figrow_master, ncols=figcol_master)

    # 選択したオリジナル画像
    gs_1 = GridSpecFromSubplotSpec(nrows=sample_num, ncols=1, subplot_spec=gs_master[0:sample_num, 0])
    for i in range(0, sample_num):
        img = V[:, sample_index_list[i]].reshape(28, 28)
        ax = fig.add_subplot(gs_1[i, :])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(img, cmap=img_cmap)

    # 棒グラフ
    gs_2 = GridSpecFromSubplotSpec(nrows=sample_num, ncols=r, subplot_spec=gs_master[0:sample_num, 1:figcol_master-1])
    for i in range(0, sample_num):
        ax = fig.add_subplot(gs_2[i, :])
        ax.axes.xaxis.set_visible(False)
        ax.bar(range(r), H[:, sample_index_list[i]])

    # 復元画像
    gs_3 = GridSpecFromSubplotSpec(nrows=sample_num, ncols=1, subplot_spec=gs_master[0:sample_num, figcol_master-1])
    for i in range(0, sample_num):
        img = reconstruct_V[:,sample_index_list[i]].reshape(28, 28)
        ax = fig.add_subplot(gs_3[i, :])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(img, cmap=img_cmap)

    # 基底画像
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=r, subplot_spec=gs_master[sample_num, 1:figcol_master-1])
    for i in range(0, r):
        ax = fig.add_subplot(gs_4[:, i])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(W_img[i], cmap=img_cmap)

    plt.show()

def show_graph(x_list, y_list, x_label, y_label, fontsize=16):
    """
    グラフを表示する．

    Parameters
    ----------
    x_list, y_list : list
        x軸のリストとy軸のリスト
    x_label, y_label : str
        x軸のラベルとy軸のラベル
    fontsize : int
        軸ラベルや目盛りのフォントサイズ
    """
    plt.rcParams["font.size"] = fontsize
    fig = plt.figure(figsize=(8.0, 6.0))
    fig.subplots_adjust(left=0.2)
    ax = fig.add_subplot(111, xlabel=x_label, ylabel=y_label)
    ax.plot(x_list, y_list)
    plt.show()

def show_graphs(x_list, y_lists, x_label, y_label, y_labels, fontsize=16):
    """
    グラフを表示する．

    Parameters
    ----------
    x_list : list
        x軸の値のリスト
    y_lists : tuple list
        y軸の値のリストのタプル
    x_label, y_label : str
        x軸のラベルとy軸のラベル
    y_labels : tuple str
        y軸の値のラベルのタプル, legendするときに使う
    fontsize : int
        軸ラベルや目盛りのフォントサイズ
    """
    plt.rcParams["font.size"] = fontsize
    fig = plt.figure(figsize=(8.0, 6.0))
    fig.subplots_adjust(left=0.2)
    ax = fig.add_subplot(111, xlabel=x_label, ylabel=y_label)
    for y_list, y_graphlabel in zip(y_lists, y_labels):
        ax.plot(x_list, y_list, label=y_graphlabel)
    ax.legend(fontsize=14)
    plt.show()

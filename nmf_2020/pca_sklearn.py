import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from modules.data import setup_mnist
from modules.array import make_baseGridImage, normalization

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

n = 28 * 28     # 画素数
sqrt_n = 28
m = 10000       # 画像数
r = 49

def main():
    V, label = setup_mnist(image_num=m)
    pca = PCA(n_components=r)
    # 基底ベクトルを取得
    W = pca.fit_transform(V)
    # これを，可視化する．棒グラフを作る．
    H = pca.components_
    print("V shape:" + str(V.shape))
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))
    
    reconstruct_V = pca.inverse_transform(W)
    print("reconstruct_V.shape: " + str(reconstruct_V.shape))
    
    # 基底ベクトルを画像のように整形
    W_image = W.T.reshape(r, 28, 28)

    # 基底画像をグリッド状に表示
    W_images = make_baseGridImage(W_image, r, normalize=True)
    img_max_abs = np.abs(W_image.max())
    img_min_abs = np.abs(W_image.min())
    colormap_range = (img_max_abs if img_max_abs > img_min_abs else img_min_abs) / 2
    sqrt_r = int(np.sqrt(r))
    plt.imshow(W_images, cmap="PiYG", extent=(0, sqrt_n * sqrt_r, 0, sqrt_n * sqrt_r))
    plt.xticks(range(0, sqrt_n * sqrt_r, sqrt_n))
    plt.yticks(range(0, sqrt_n * sqrt_r, sqrt_n))
    plt.grid(which = "major", color = "black", alpha = 0.8, linestyle = "--", linewidth = 1)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.show()

    '''
    # オリジナル画像と再構成画像
    sample_num = 5
    sample_index_list = np.random.randint(0, m, size=sample_num)
    fig = plt.figure()
    for i in range(0, sample_num):
        img = V[:,sample_index_list[i]].reshape(28, 28)
        ax = fig.add_subplot(2,5,i+1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        aximg = ax.imshow(img, cmap="PiYG", vmin=-1, vmax=1)
        fig.colorbar(aximg, ax=ax)
        img = reconstruct_V[:,sample_index_list[i]].reshape(28, 28)
        ax = fig.add_subplot(2,5,i+6)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        aximg = ax.imshow(img, cmap="PiYG", vmin=-1, vmax=1)
        fig.colorbar(aximg, ax=ax)
    plt.show()
    '''


    '''
    # ここからグラフ作成(棒グラフで可視化)
    sample_num = 5
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
        ax.imshow(img, cmap="PiYG", vmin=-1, vmax=1)

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
        ax.imshow(img, cmap="PiYG", vmin=-1, vmax=1)

    # 基底画像
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=r, subplot_spec=gs_master[sample_num, 1:figcol_master-1])
    for i in range(0, r):
        ax = fig.add_subplot(gs_4[:, i])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        img_max_abs = np.abs(W_image[i].max())
        img_min_abs = np.abs(W_image[i].min())
        colormap_range = img_max_abs if img_max_abs > img_min_abs else img_min_abs
        ax.imshow(W_image[i], cmap="PiYG", vmin=-colormap_range, vmax=colormap_range)

    plt.show()
    '''

if __name__ == "__main__":
    main()
    
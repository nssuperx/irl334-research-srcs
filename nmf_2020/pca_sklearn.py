import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from modules.data_function import setup_mnist

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

# HACK:変数名がまずい

n = 28 * 28     # 画素数
m = 10000       # 画像数
r = 100

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

    sample_num = 5
    sample_index_list = np.random.randint(0, m, size=sample_num)
    fig = plt.figure()
    for i in range(0, sample_num):
        img = V[:,sample_index_list[i]].reshape(28, 28)
        ax = fig.add_subplot(2,5,i+1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(img)
        img = reconstruct_V[:,sample_index_list[i]].reshape(28, 28)
        ax = fig.add_subplot(2,5,i+6)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(img)
    plt.show()

    '''
    # ここからグラフ作成
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
        ax.imshow(img)

    # 棒グラフ
    gs_2 = GridSpecFromSubplotSpec(nrows=sample_num, ncols=r, subplot_spec=gs_master[0:sample_num, 1:figcol_master-1])
    for i in range(0, sample_num):
        ax = fig.add_subplot(gs_2[i, :])
        ax.axes.xaxis.set_visible(False)
        ax.bar(range(r), H[:, sample_index_list[i]])

    # 復元画像
    # FIXME:ここ全然違う．復習しないといけない．
    gs_3 = GridSpecFromSubplotSpec(nrows=sample_num, ncols=1, subplot_spec=gs_master[0:sample_num, figcol_master-1])
    for i in range(0, sample_num):
        img = np.dot(W, H[:, sample_index_list[i]]).reshape(28, 28)
        ax = fig.add_subplot(gs_3[i, :])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(img)

    # 基底画像
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=r, subplot_spec=gs_master[sample_num, 1:figcol_master-1])
    for i in range(0, r):
        ax = fig.add_subplot(gs_4[:, i])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(W_image[i])

    plt.show()
    '''
    
if __name__ == "__main__":
    main()
    
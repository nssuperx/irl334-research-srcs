import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

n = 28 * 28 # 画素数
m = 10000    # 画像数
r = 10

def main():
    V, V_test, label, label_test = setup_mnist()
    nmf = NMF(n_components=r, max_iter=400)
    # 基底ベクトルを取得
    W = nmf.fit_transform(V)
    # これを，可視化する．棒グラフを作る．
    H = nmf.components_
    print("W shape:" + str(W.shape))
    print("V shape:" + str(V.shape))
    print("H shape:" + str(H.shape))
    print("iter:%d" % (nmf.n_iter_))
    
    # 基底ベクトルを画像のように整形
    W_image = W.T.reshape(r, 28, 28)

    # ここからグラフ作成
    sample_num = 10
    sample_list = np.random.randint(0, m, size=sample_num)
    figrow_master = sample_num + 1
    figcol_master = r + 1
    fig = plt.figure(figsize=(figrow_master, figcol_master))
    gs_master = GridSpec(nrows=figrow_master, ncols=figcol_master)
    gs_1 = GridSpecFromSubplotSpec(nrows=sample_num, ncols=1, subplot_spec=gs_master[0:sample_num, 0])
    for i in range(0, sample_num):
        img = V[:, sample_list[i]].reshape(28, 28)
        ax = fig.add_subplot(gs_1[i, :])
        ax.imshow(img)

    gs_2 = GridSpecFromSubplotSpec(nrows=sample_num, ncols=r, subplot_spec=gs_master[0:sample_num, 1:figcol_master])
    for i in range(0, sample_num):
        ax = fig.add_subplot(gs_2[i, :])
        ax.bar(range(r), H[:, sample_list[i]])

    gs_3 = GridSpecFromSubplotSpec(nrows=1, ncols=r, subplot_spec=gs_master[sample_num, 1:figcol_master])
    for i in range(0, r):
        ax = fig.add_subplot(gs_3[:, i])
        ax.imshow(W_image[i])
    plt.show()
    
    # 基底を作るときに使ってない画像は復元できない？NMFの弱点？


def setup_mnist():
    """
    sklearnでmnistのデータを作り，numpy配列を作る．
    """
    digits = fetch_openml(name='mnist_784', version=1, data_home="mnist")
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=m, random_state=0, shuffle=True)

    return x_train.T, x_test.T, y_train, y_test

if __name__ == "__main__":
    main()
    
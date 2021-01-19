import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

# http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

# ValueError: Key backend: 'qtagg' is not a valid value for backend; supported values are
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg',
# 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

n = 28 * 28     # 画素数
m = 10000       # 画像数
r = 20          # 基底数

def main():
    V, V_test, label, label_test = setup_mnist()
    # nmf = NMF(n_components=r, max_iter=10000, beta_loss='frobenius', solver='cd', tol=0.0001, random_state=0)
    nmf = NMF(n_components=r, max_iter=10000, beta_loss='kullback-leibler', solver='mu', tol=0.0001, random_state=0)
    # 基底ベクトルを取得
    W = nmf.fit_transform(V)
    # これを，可視化する．棒グラフを作る．
    H = nmf.components_
    print("V shape:" + str(V.shape))
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))
    print("iter:%d" % (nmf.n_iter_))
    
    # TODO: 画像表示処理を関数化

    # 基底ベクトルを画像のように整形
    W_image = W.T.reshape(r, 28, 28)

    # 基底画像の確認
    show_W_img_num = 10
    fig = plt.figure()
    for i in range(show_W_img_num):
        ax = fig.add_subplot(2,5,i+1)
        ax.imshow(W_image[i])
    plt.show()

    """
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
    """
    

def setup_mnist():
    """
    sklearnでmnistのデータを作り，numpy配列を作る．
    """
    digits = fetch_openml(name='mnist_784', version=1, data_home="mnist")
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=m, random_state=0, shuffle=True)

    return x_train.T, x_test.T, y_train, y_test

if __name__ == "__main__":
    main()
    
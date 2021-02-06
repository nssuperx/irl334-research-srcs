import numpy as np

def normalize(V, axis=None):
    """
    行列の要素を[0, 1]にしたものを返す．

    Parameters
    ----------
    V: numpy.adarray
        行列
    axis: int
        方向

    Return
    ----------
    numpy.adarray
        確率分布になった行列
    """

    min_val = V.min(axis=axis, keepdims=True)
    max_val = V.max(axis=axis, keepdims=True)
    return (V - min_val)/(max_val - min_val)
    

def prob_dist(V):
    """
    行列をaxis=0の方向で確率分布の行列にする．

    Parameters
    ----------
    V: numpy.adarray
        行列

    Return
    ----------
    numpy.adarray
        確率分布になった行列
    """
    return V / np.tile(np.sum(V, axis=0), (V.shape[0], 1))

def make_baseGridImage(W, h_num, v_num, img_normalize=False):
    W_list = []
    for i in range(v_num):
        h_list = []
        for j in range(h_num):
            if img_normalize:
                h_list.append(normalize(W[(i * h_num) + j]))
            else:
                h_list.append(W[(i * h_num) + j])
        W_list.append(h_list)

    return np.block(W_list)

def make_baseGridImage_square(W, r, img_normalize=False):
    sqrt_r = int(np.sqrt(r))
    return make_baseGridImage(W, sqrt_r, sqrt_r, img_normalize)

def search_near_imagepair(W, V):
    """
    ※未完成
    類似度が近い画像のペアを見つける

    Parameters
    ----------
    W, V: numpy.adarray
        行列
    """
    # ローカルミニマムに落ちた基底画像と近いものを探す機能を作っていたが，断念．
    # Hを見ていったほうがいいと思う．
    min_v_idx_list = []

    nW = normalize(W, axis=0)
    nV = normalize(V, axis=0)

    for i in range(W.shape[1]):
        w = nW[:,i].reshape(W.shape[0], 1)
        tmp = (np.tile(w, (1, V.shape[1])) - nV)
        diff = np.power(np.sum(tmp, axis=0), 2)
        min_v_idx_list.append(np.copy(np.argmin(diff)))

    print(min_v_idx_list)

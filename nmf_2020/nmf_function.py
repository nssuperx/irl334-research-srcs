import numpy as np

def kl_divergence(V, W, H):
    WH = np.dot(W, H) + epsilon
    log_WH = np.log(WH)
    F = np.sum(np.multiply(V, log_WH) - WH)
    return F

def frobenius_norm(V, W, H):
    pass

'''
\mu番目の「例題」 $\mu = 1,2, ..., m$
i番目の「ピクセル」 $i = 1,2, ..., n$
a番目の「基底」 $a = 1,2, ..., r$
'''
# 値更新
def update(V, W, H):
    """
    NMF更新
    Lee & Seung アルゴリズム

    Parameters
    ----------
    V: numpy.adarray
        オリジナルのデータ
    W: numpy.adarray
        基底
    H: numpy.adarray
        重み

    Returns
    --------
    W: numpy.adarray
        基底
    H: numpy.adarray
        重み
    """
    WH = np.dot(W, H) + epsilon
    W = W * np.dot(V / WH, H.T)

    W_tmp = np.sum(W, axis=0)
    W = W / np.tile(W_tmp, (n, 1))

    WH = np.dot(W, H) + epsilon
    H = H * np.dot(W.T, (V/WH))

    return W, H
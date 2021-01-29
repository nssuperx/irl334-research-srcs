import numpy as np

def normalization(V):
    return (V - V.min()) / (V.max() - V.min())

def make_baseGridImage_square(W, r, normalize=False):
    sqrt_r = int(np.sqrt(r))
    W_list = []
    for i in range(sqrt_r):
        h_list = []
        for j in range(sqrt_r):
            if normalize:
                h_list.append(normalization(W[(i * sqrt_r) + j]))
            else:
                h_list.append(W[(i * sqrt_r) + j])
        W_list.append(h_list)

    return np.block(W_list)

def make_baseGridImage(W, h_num, v_num, normalize=False):
    W_list = []
    for i in range(v_num):
        h_list = []
        for j in range(h_num):
            if normalize:
                h_list.append(normalization(W[(i * h_num) + j]))
            else:
                h_list.append(W[(i * h_num) + j])
        W_list.append(h_list)

    return np.block(W_list)

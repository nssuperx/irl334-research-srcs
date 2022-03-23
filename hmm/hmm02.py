import math
import random
import numpy as np
import matplotlib.pyplot as plt

from hmm_class import HMM

def hmm02():
    n = 200
    sigma = 0.7
    hmm_buf = [HMM(n, sigma) for i in range(10)]
    x_transition = [0 for i in range(10)]

    t = range(n)

    # 10通りつくる
    for i in range(10):
        hmm_buf[i].generate_x()

    # たくさん遷移してるのを探す
    # ハミング距離を計算
    for i in range(10):
        x_transition[i] = np.sum(np.absolute(hmm_buf[i].x[0:n-2] - hmm_buf[i].x[1:n-1]))

    # ここから実験
    hmm = hmm_buf[x_transition.index(max(x_transition))]
    hmm.generate_y()

    t = range(n)
    plt.plot(t, hmm.x, label='x')
    plt.plot(t, hmm.y, '.g', label='y') # g は緑色， * は点

    plt.title('Make y_obs') 
    plt.xlabel('t') # X 軸
    plt.ylabel('x, y') # Y 軸
    plt.legend()

    plt.show() # 描画

if __name__ == '__main__':
    hmm02()
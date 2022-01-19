import math
import random
import numpy as np
import matplotlib.pyplot as plt

from hmm_class import HMM

def hmm04():
    n = 200
    sigma = 0.7
    hmm_buf = [HMM(n, sigma) for i in range(10)]
    x_transition = [0 for i in range(10)]

    t = range(n)

    #10通りつくる
    for i in range(10):
        hmm_buf[i].generate_x()

    # たくさん遷移してるのを探す
    # ハミング距離を計算
    for i in range(10):
        x_transition[i] = np.sum(np.absolute(hmm_buf[i].x[0:n-2] - hmm_buf[i].x[1:n-1]))

    # ここから実験
    hmm = hmm_buf[x_transition.index(max(x_transition))]
    hmm.generate_y()
    hmm.compute_xmap()
    hamming_distanse = calc_hamming(hmm)

    print('hamming_distanse = ' + str(hamming_distanse))
        
    t = range(n)
    plt.plot(t, hmm.x, label='Xorg')
    plt.plot(t, hmm.xmap + 3, label='Xmap')
    plt.plot(t, hmm.y, '.g', label='y') # g は緑色， * は点

    plt.title('Original Signal, Observations, Mapping Signal') 
    plt.xlabel('t') # X 軸
    plt.ylabel('x, y') # Y 軸
    plt.legend() # 描画

    plt.show() # 描画

def calc_hamming(hmm: HMM) -> int:
    return np.sum(np.absolute(hmm.x - hmm.xmap))


if __name__ == '__main__':
    hmm04()
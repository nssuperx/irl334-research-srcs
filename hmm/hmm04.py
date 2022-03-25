import math
import random
import numpy as np
import matplotlib.pyplot as plt

from hmm_class import HMM

def hmm04():
    n = 200
    sigma = 0.7
    sample_num = 10
    hmm_buf = [HMM(n, sigma, 0.99, 0.97) for i in range(10)]
    x_transition = np.empty(sample_num)

    #10通りつくる
    for i in range(sample_num):
        hmm_buf[i].generate_x()

    # たくさん遷移してるのを探す
    # ハミング距離を計算
    for i in range(sample_num):
        x_transition[i] = np.sum(np.absolute(hmm_buf[i].x[0:n-2] - hmm_buf[i].x[1:n-1]))

    # ここから実験
    hmm = hmm_buf[x_transition.argmax()]
    hmm.generate_y()
    hmm.compute_xmap()
    hamming_distanse = hmm.calc_error()

    print('hamming_distanse = ' + str(hamming_distanse))
        
    t = range(n)
    plt.plot(t, hmm.x, label='Xorg')
    plt.plot(t, hmm.xmap + 3, label='Xmap')
    plt.plot(t, hmm.y, '.g', label='y') # g は緑色， * は点

    plt.title('Original Signal, Observations, Mapping Signal') 
    plt.xlabel('t') # X 軸
    plt.ylabel('x, y') # Y 軸
    plt.legend()

    plt.show() # 描画


if __name__ == '__main__':
    hmm04()
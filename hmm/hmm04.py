# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt

from hmm_class import HMM

def hmm04():
    n = 200
    sigma = 0.7
    hmm_buf = [HMM(n,sigma)] * 10
    x_transition = [0] * 10

    t = range(n)

    #10通りつくる
    for i in range(10):
        hmm_buf[i].generate_x()

    #たくさん遷移してるのを探す
    for i in range(10):
        for j in range(1,n):
            x_transition[i] += abs(hmm_buf[i].x[j-1] - hmm_buf[i].x[j])

    #そのインデックスを保持
    Xindex = x_transition.index(max(x_transition))

    #見た目をきれいにするために、新しいインスタンスを作成
    hmm = hmm_buf[Xindex]
    hmm.generate_y()
    hmm.compute_xmap()
    hamming_distanse = calc_hamming(hmm)

    for i in range(0,n):
        hmm.xmap[i] = hmm.xmap[i]+3

    print('hamming_distanse = ' + str(hamming_distanse))
        
    t = range(n)
    plt.plot(t, hmm.x, label='Xorg')
    plt.plot(t, hmm.xmap, label='Xmap')
    plt.plot(t, hmm.y, '.g', label='y') # g は緑色， * は点

    plt.title('Original Signal, Observations, Mapping Signal') 
    plt.xlabel('t') # X 軸
    plt.ylabel('x, y') # Y 軸
    plt.legend() # 描画

    plt.show() # 描画

def calc_hamming(hmm):
    hamming_distanse = 0
    for i in range(hmm.n):
        hamming_distanse += abs(hmm.x[i] - hmm.xmap[i])
    return hamming_distanse


if __name__ == '__main__':
    hmm04()
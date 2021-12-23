# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt

from hmm_class import HMM

def hmm02():
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

    t = range(n)
    plt.plot(t, hmm.x, label='x')
    plt.plot(t, hmm.y, '.g', label='y') # g は緑色， * は点

    plt.title('Make y_obs') 
    plt.xlabel('t') # X 軸
    plt.ylabel('x, y') # Y 軸
    plt.legend() # 描画

    plt.show() # 描画

if __name__ == '__main__':
    hmm02()
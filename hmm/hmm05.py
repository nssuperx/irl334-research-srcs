import math
import random
import numpy as np
import matplotlib.pyplot as plt

from hmm_class import HMM

hmm_num = 1000

def hmm05():
    n = 200
    sigma = 0.7
    hmm = [HMM(n, sigma) for i in range()]
    hamming_distanse = [0 for i in range()]

    #hmm_num通りつくる
    for i in range(hmm_num):
        hmm[i].generate_x()
        hmm[i].generate_y()
        hmm[i].compute_xmap()
        hamming_distanse[i] = calc_hamming(hmm[i])

    hd_mean = np.mean(hamming_distanse)
    hd_var = np.var(hamming_distanse)

    print('hamming_distanse_mean: ' + str(hd_mean))
    print('hamming_distanse_var: ' + str(hd_var))

    plt.hist(hamming_distanse, ec='black',color='orange')

    plt.title('hamming distanse hist') 
    #plt.legend() # 描画

    plt.show() # 描画

def calc_hamming(hmm: HMM) -> int:
    return np.sum(np.absolute(hmm.x - hmm.xmap))


if __name__ == '__main__':
    hmm05()
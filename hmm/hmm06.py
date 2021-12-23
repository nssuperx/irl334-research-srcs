# -*- coding: utf-8 -*-
import math
import random
import numpy
import matplotlib.pyplot as plt

from hmm_class import HMM

n = 200
hmm_num = 1000
sigma_first = 0.3
sigma_last = 2.0
sigma_step = 0.1

def hmm06():
    sigma_list = make_sigma_list()
    mean_list = []
    std_list_up = []
    std_list_down = []

    for i in sigma_list:
        buffer = task(i)
        mean_list.append(buffer[0])
        std_list_up.append(buffer[0] + buffer[1])
        std_list_down.append(buffer[0] - buffer[1])

    plt.plot(sigma_list, mean_list, label='mean')
    plt.plot(sigma_list, std_list_up, label='std up')
    plt.plot(sigma_list, std_list_down, label='std down')

    plt.title('hamming distanse') 
    plt.xlabel('sigma')
    plt.ylabel('y')
    plt.legend() # 描画

    plt.show() # 描画

def calc_hamming(hmm):
    hamming_distanse = 0
    for i in range(hmm.n):
        hamming_distanse += abs(hmm.x[i] - hmm.xmap[i])
    return hamming_distanse

def make_sigma_list():
    sigma_list = []
    sigma = sigma_first
    while sigma <= sigma_last:
        sigma_list.append(sigma)
        sigma += sigma_step
        sigma = round(sigma,1)
    
    return sigma_list

def task(sigma):
    hmm = [HMM(n,sigma)] * hmm_num
    hamming_distanse = [0] * hmm_num

    #hmm_num通りつくる
    for i in range(hmm_num):
        hmm[i].generate_x()
        hmm[i].generate_y()
        hmm[i].compute_xmap()
        hamming_distanse[i] = calc_hamming(hmm[i])

    hd_mean = numpy.mean(hamming_distanse)
    hd_std = numpy.std(hamming_distanse)

    return hd_mean, hd_std


if __name__ == '__main__':
    hmm06()
    #print(make_sigma_list())
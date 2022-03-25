import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmm_class import HMM

n = 200
hmm_num = 1000
sigma_first = 0.3
sigma_last = 2.0
sigma_step = 0.1

def hmm06() -> None:
    sigma_list = np.arange(sigma_first, sigma_last, sigma_step)
    mean_list = []
    std_list_up = []
    std_list_down = []

    for i in tqdm(sigma_list):
        mean, std = task(i)
        mean_list.append(mean)
        std_list_up.append(mean + std)
        std_list_down.append(mean - std)

    plt.plot(sigma_list, mean_list, label='mean')
    plt.plot(sigma_list, std_list_up, label='std+')
    plt.plot(sigma_list, std_list_down, label='std-')

    plt.title('hamming distanse') 
    plt.xlabel('sigma')
    plt.ylabel('y')
    plt.legend()

    plt.show() # 描画

def task(sigma: float) -> tuple[float, float]:
    hamming_distanse = np.empty(hmm_num)

    #hmm_num通りつくる
    for i in tqdm(range(hmm_num), leave=False):
        hmm = HMM(n, sigma, 0.99, 0.97)
        hmm.generate_x()
        hmm.generate_y()
        hmm.compute_xmap()
        hamming_distanse[i] = hmm.calc_error()

    return np.mean(hamming_distanse), np.std(hamming_distanse)


if __name__ == '__main__':
    hmm06()

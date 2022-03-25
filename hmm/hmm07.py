import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmm_class import HMM, Model

n = 200
hmm_num = 1000
theta_first = 0.7
theta_last = 0.99
theta_step = 0.05

def hmm07() -> None:
    sigma = 0.7
    theta_list = np.arange(theta_first, theta_last, theta_step)
    mean_list = []
    std_list_up = []
    std_list_down = []

    # おおもとのxとyを生成．固定しておく．
    hmm = HMM(n, sigma, 0.99, 0.97)
    hmm.generate_x()
    hmm.generate_y()

    for theta in tqdm(theta_list):
        mean, std = task(hmm, theta)
        mean_list.append(mean)
        std_list_up.append(mean + std)
        std_list_down.append(mean - std)

    plt.plot(theta_list, mean_list, label='mean')
    plt.plot(theta_list, std_list_up, label='std+')
    plt.plot(theta_list, std_list_down, label='std-')

    plt.title('hamming distanse') 
    plt.xlabel('theta')
    plt.ylabel('y')
    plt.xticks(np.arange(min(theta_list), max(theta_list) + 0.01, theta_step))
    plt.legend()

    plt.show() # 描画

def task(hmm: HMM, theta: float) -> tuple[float, float]:
    hamming_distanse = np.empty(hmm_num)
    
    for i in tqdm(range(hmm_num), leave=False):
        # 変更するのはxmapを計算するときに使うモデルのみ
        hmm.model_xmap = Model(theta, theta)
        hmm.compute_xmap()
        hamming_distanse[i] = hmm.calc_error()

    return np.mean(hamming_distanse), np.std(hamming_distanse)


if __name__ == '__main__':
    hmm07()

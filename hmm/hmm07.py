from typing import List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmm_class import HMM, Model

n = 200
sigma = 0.7
trial_num = 1000
theta_first = 0.900
theta_last = 0.999
theta_step = 0.001

def hmm07() -> None:
    theta_list = np.arange(theta_first, theta_last, theta_step)
    error_list = []

    for i in tqdm(range(trial_num)):
        error = task(theta_list)
        error_list.append(error)
    
    error_array = np.array(error_list)
    mean_array = error_array.mean(axis=0, dtype=np.float64)
    std_array = error_array.std(axis=0, dtype=np.float64)

    plt.plot(theta_list, mean_array, label='mean')
    plt.plot(theta_list, mean_array + std_array, label='std+')
    plt.plot(theta_list, mean_array - std_array, label='std-')

    plt.title('hamming distanse') 
    plt.xlabel('theta')
    plt.ylabel('error')
    # plt.xticks(np.arange(min(theta_list), max(theta_list) + 0.01, theta_step))
    plt.legend()

    plt.show() # 描画

def task(theta_list: np.ndarray) -> List[int]:
    error: List[float] = []
    hmm = HMM(n, sigma, 0.99, 0.97)
    hmm.generate_x()
    hmm.generate_y()
    for theta in tqdm(theta_list, leave=False):
        hmm.model_xmap = Model(theta, theta)
        hmm.compute_xmap()
        error.append(hmm.calc_error())

    return error


if __name__ == '__main__':
    hmm07()

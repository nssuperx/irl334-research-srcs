from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from itertools import product
from logging import getLogger, StreamHandler, INFO, Formatter, CRITICAL
import time

from hmm_class import HMM

# logger setting
logger = getLogger()
handler = StreamHandler()
handler.setFormatter(Formatter('%(asctime)s [%(levelname)s]: %(message)s'))
handler.setLevel(INFO)
logger.addHandler(handler)
logger.setLevel(INFO)

@dataclass
class trial_result:
    p00: float
    p11: float
    hamming_distanse: float

def hmm_task(n: int, sigma: float, p00: float, p11: float, trial_num: int) -> trial_result:
    start_time = time.time()
    hamming_distanse_list = []
    hmm = HMM(n, sigma, 0.97, 0.99, p00, p11)
    for i in range(trial_num):
        hmm.generate_x()
        hmm.generate_y()
        hmm.compute_xmap()
        hamming_distanse_list.append(hmm.calc_error())
    hamming_mean = np.asarray(hamming_distanse_list, dtype=np.float64).mean()
    logger.info(f'[p00={p00:.2f}, p11={p11:.2f}] process time: {time.time() - start_time}')
    return trial_result(p00, p11, hamming_mean)

def main():
    n = 200
    sigma = 0.7
    results: List[trial_result] = []

    p_args = list(product(np.arange(0.50, 1.00, 0.01), np.arange(0.50, 1.00, 0.01)))
    hmm_args = [(n, sigma, arg[0], arg[1], 1000) for arg in p_args]
    with Pool(processes=os.cpu_count()//2) as pool:
        results = pool.starmap(hmm_task, hmm_args)
    logger.info(f'p_args len: {len(results)}')

    array_x = np.zeros((len(results)))
    array_y = np.zeros((len(results)))
    array_z = np.zeros((len(results)))
    for i, r in enumerate(results):
        array_x[i] = r.p00
        array_y[i] = r.p11
        array_z[i] = r.hamming_distanse

    np.savez('./hmm_results_mp.npz', x=array_x, y=array_y, z=array_z)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(array_x,array_y,array_z)
    plt.show()

if __name__ == '__main__':
    main()

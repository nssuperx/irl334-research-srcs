import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmm_class import HMM

hmm_num = 1000

def hmm05():
    n = 200
    sigma = 0.7
    hamming_distanse = [0 for i in range(hmm_num)]

    #hmm_num通りつくる
    for i in tqdm(range(hmm_num)):
        hmm = HMM(n, sigma)
        hmm.generate_x()
        hmm.generate_y()
        hmm.compute_xmap()
        hamming_distanse[i] = hmm.calc_hamming()

    hd_mean = np.mean(hamming_distanse)
    hd_var = np.var(hamming_distanse)

    print('hamming_distanse_mean: ' + str(hd_mean))
    print('hamming_distanse_var: ' + str(hd_var))

    plt.hist(hamming_distanse, ec='black',color='orange')

    plt.title('hamming distanse hist') 
    #plt.legend()

    plt.show() # 描画


if __name__ == '__main__':
    hmm05()
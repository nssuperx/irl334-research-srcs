from typing import List
import numpy as np
import matplotlib.pyplot as plt

from hmm_class import HMM

class trial_result:
    def __init__(self, p00: float, p11: float, hamming_distance: float) -> None:
        self.p00 = p00
        self.p11 = p11
        self.hamming_distanse = hamming_distance

def main():
    n = 200
    sigma = 0.7
    results: List[trial_result] = []

    for p00 in np.arange(0.50, 1.00, 0.01):
        for p11 in np.arange(0.50, 1.00, 0.01):
            hamming_distanse = []
            for i in range(1000):
                hmm = HMM(n, sigma, 0.97, 0.99, p00, p11)
                hmm.generate_x()
                hmm.generate_y()
                hmm.compute_xmap()
                hamming_distanse.append(calc_hamming(hmm))
            hamming_mean = np.asarray(hamming_distanse, dtype=np.float64).mean()
            # print(f'p00={p00:.2f}, p11={p11:.2f}: hamming_distance mean = {hamming_mean}')
            results.append(trial_result(p00, p11, hamming_mean))

    array_x = np.zeros((len(results)))
    array_y = np.zeros((len(results)))
    array_z = np.zeros((len(results)))
    for i, r in enumerate(results):
        array_x[i] = r.p00
        array_y[i] = r.p11
        array_z[i] = r.hamming_distanse

    np.savez('./hmm_results.npz', x=array_x, y=array_y, z=array_z)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(array_x,array_y,array_z)
    plt.show()

def calc_hamming(hmm: HMM) -> int:
    return np.sum(np.absolute(hmm.x - hmm.xmap))

# 変化が多いモデルを返す
# 基本は使用しない
def prepare_model(n: int, sigma: float) -> HMM:
    x_transition = [0 for i in range(10)]
    hmm_buf = [HMM(n, sigma, 0.97, 0.97, 0.97, 0.97) for i in range(10)]
    #10通りつくる
    for i in range(10):
        hmm_buf[i].generate_x()

    # たくさん遷移してるのを探す
    # ハミング距離を計算
    for i in range(10):
        x_transition[i] = np.sum(np.absolute(hmm_buf[i].x[0:n-2] - hmm_buf[i].x[1:n-1]))

    return hmm_buf[x_transition.index(max(x_transition))]


if __name__ == '__main__':
    main()

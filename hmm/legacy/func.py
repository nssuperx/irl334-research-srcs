import numpy as np
from ..hmm_class import HMM

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

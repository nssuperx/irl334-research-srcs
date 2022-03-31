import math
import numpy as np
import hmm_func

# speed hmm07.py
# cython: elapsed_time: 270.92019057273865 sec
# normal: elapsed_time: 312.12768054008484 sec
# 
# n = 200
# sigma = 0.7
# trial_num = 1000
# theta_first = 0.900
# theta_last = 0.999
# theta_step = 0.001

class Model:
    def __init__(self, p00: float, p11: float) -> None:
        self.p00 = p00
        self.p01 = 1.00 - p00
        self.p10 = 1.00 - p11
        self.p11 = p11


class HMM:
    def __init__(self, n: int, sigma: float, x_p00: float, x_p11: float, xmap_p00: float = None, xmap_p11: float = None) -> None:
        self.n = n
        self.sigma = sigma

        self.x = np.zeros(self.n, dtype=np.int8)
        self.xmap = np.zeros(self.n, dtype=np.int8)
        self.y = np.zeros(self.n, dtype=np.float32)

        self.model_x = Model(x_p00, x_p11)
        if (xmap_p00 is None or xmap_p11 is None):
            self.model_xmap = Model(x_p00, x_p11)
        else:
            self.model_xmap = Model(xmap_p00, xmap_p11)

    def generate_x(self) -> None:
        self.x = hmm_func.generate_x(self.n, self.model_x.p00, self.model_x.p11)
        
    def generate_y(self) -> None:
        self.y = np.random.normal(self.x, self.sigma)

    def compute_xmap(self) -> None:
        self.xmap = hmm_func.compute_xmap(self.n, self.sigma, self.y, self.model_xmap.p00, self.model_xmap.p01, self.model_xmap.p10, self.model_xmap.p11)

    def calc_error(self) -> int:
        return np.sum(np.absolute(self.x - self.xmap))

import math
import numpy as np

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
        self.S = np.zeros((self.n, 2), dtype=np.int8)        # 次のをみて、Xiが取るべき値
        self.C = np.zeros((self.n, 2), dtype=np.float32)     # もっともらしさ

        self.x = np.zeros(self.n, dtype=np.int8)
        self.xmap = np.zeros(self.n, dtype=np.int8)
        self.y = np.zeros(self.n, dtype=np.float32)

        self.model_x = Model(x_p00, x_p11)
        if (xmap_p00 is None or xmap_p11 is None):
            self.model_xmap = Model(x_p00, x_p11)
        else:
            self.model_xmap = Model(xmap_p00, xmap_p11)

    def generate_x(self) -> None:
        self.x[0] = 0 if np.random.random() < 0.5 else 1

        for i in range(1,self.n):
            r = np.random.random()
            if ( self.x[i-1] == 0 ):
                self.x[i] = 0 if r < self.model_x.p00 else 1
            else:
                self.x[i] = 1 if r < self.model_x.p11 else 0
        
    def generate_y(self) -> None:
        self.y = np.random.normal(self.x, self.sigma)

    def compute_xmap(self) -> None:
        log_p00 = math.log(self.model_xmap.p00)
        log_p01 = math.log(self.model_xmap.p01)
        log_p10 = math.log(self.model_xmap.p10)
        log_p11 = math.log(self.model_xmap.p11)

        log_sigma = math.log(self.sigma)

        self.C[0][0] = -pow(self.y[0] - 0.0, 2.0) / (2.0*pow(self.sigma, 2.0)) - log_sigma
        self.C[0][1] = -pow(self.y[0] - 1.0, 2.0) / (2.0*pow(self.sigma, 2.0)) - log_sigma

        for i in range(1,self.n):
            self.C[i][0] = -pow((self.y[i] - 0.0), 2.0)/ (2.0*pow(self.sigma, 2.0)) - log_sigma
            if(self.C[i-1][0] + log_p00 > self.C[i-1][1] + log_p10):
                self.S[i-1][0] = 0
                self.C[i][0] += self.C[i-1][0] + log_p00
            else:
                self.S[i-1][0] = 1
                self.C[i][0] += self.C[i-1][1] + log_p10

            self.C[i][1] = -pow((self.y[i] - 1.0), 2.0)/ (2.0*pow(self.sigma, 2.0)) - log_sigma
            if(self.C[i-1][0] + log_p01 > self.C[i-1][1] + log_p11):
                self.S[i-1][1] = 0
                self.C[i][1] += self.C[i-1][0] + log_p01
            else:
                self.S[i-1][1] = 1
                self.C[i][1] += self.C[i-1][1] + log_p11
        
        self.xmap[self.n-1] = 0 if self.C[self.n-1][0] > self.C[self.n-1][1] else 1

        for i in range(2,self.n+1):
            self.xmap[self.n-i] = self.S[self.n-i][self.xmap[self.n-i+1]]

    def calc_error(self) -> int:
        return np.sum(np.absolute(self.x - self.xmap))

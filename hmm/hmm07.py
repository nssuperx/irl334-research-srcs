import math
import random
import numpy as np
import matplotlib.pyplot as plt

n = 200
hmm_num = 1000
theta_first = 0.7
theta_last = 1.0
theta_step = 0.05

class HMM:
    def __init__(self, n, sigma, p00, p11):
        self.n = n
        self.sigma = sigma
        self.S = np.zeros((self.n, 2), dtype=np.int8)     #次のをみて、Xiが取るべき値
        self.C = np.zeros((self.n, 2), dtype=np.float32)     #もっともらしさ

        self.p00 = p00
        self.p01 = 1.00 - p00
        self.p10 = 1.00 - p11
        self.p11 = p11

        self.x = np.zeros(self.n, dtype=np.int8)
        self.xmap = np.zeros(self.n, dtype=np.int8)
        self.y = np.zeros(self.n, dtype=np.float32)
        
        self.T = 2*pow(sigma,2)

        self.log_p00 = math.log(self.p00)
        self.log_p01 = math.log(self.p01)
        self.log_p10 = math.log(self.p10)
        self.log_p11 = math.log(self.p11)


    def generate_x(self):
        if (random.random() < 0.5):
            self.x[0] = 0
        else:
            self.x[0] = 1

        for i in range(1,self.n):
            r = random.random()
            if ( self.x[i-1] == 0 ):
                if ( r < self.p00 ):
                    self.x[i] = 0
                else:
                    self.x[i] = 1
            else:
                if ( r < self.p11 ):
                    self.x[i] = 1
                else:
                    self.x[i] = 0
        
    def generate_y(self):
        for i in range(0,self.n):
            self.y[i] = random.gauss(self.x[i],self.sigma)

    def compute_xmap(self):
        self.C[0][0] = -pow(self.y[0] - 0, 2) / self.T
        self.C[0][1] = -pow(self.y[0] - 1, 2) / self.T

        for i in range(1,self.n):
            self.C[i][0] = -pow(self.y[i] - 0, 2) / self.T
            self.C[i][1] = -pow(self.y[i] - 1, 2) / self.T
            t00 = self.C[i-1][0] + self.C[i][0] + self.log_p00
            t01 = self.C[i-1][0] + self.C[i][1] + self.log_p01
            t10 = self.C[i-1][1] + self.C[i][0] + self.log_p10
            t11 = self.C[i-1][1] + self.C[i][1] + self.log_p11

            if t00 > t10:
                self.S[i-1][0] = 0
                self.C[i][0] += self.C[i-1][0] + math.log(0.99)
            else:
                self.S[i-1][0] = 1
                self.C[i][0] += self.C[i-1][1] + math.log(0.03)

            if t01 > t11:
                self.S[i-1][1] = 0
                self.C[i][1] += self.C[i-1][0] + math.log(0.01)
            else:
                self.S[i-1][1] = 1
                self.C[i][1] += self.C[i-1][1] + math.log(0.97)
        
        self.xmap[self.n-1] = 0 if self.C[self.n-1][0] > self.C[self.n-1][1] else 1

        for i in range(2,self.n+1):
            self.xmap[self.n-i] = self.S[self.n-i][self.xmap[self.n-i+1]]


def hmm07():
    sigma = 0.7
    theta_list = make_theta_list()
    mean_list = []
    std_list_up = []
    std_list_down = []

    for i in theta_list:
        buffer = task(sigma, i)
        mean_list.append(buffer[0])
        std_list_up.append(buffer[0] + buffer[1])
        std_list_down.append(buffer[0] - buffer[1])

    plt.plot(theta_list, mean_list, label='mean')
    plt.plot(theta_list, std_list_up, label='std up')
    plt.plot(theta_list, std_list_down, label='std down')

    plt.title('hamming distanse') 
    plt.xlabel('theta')
    plt.ylabel('y')
    plt.xticks(np.arange(min(theta_list), max(theta_list) + 0.01, theta_step))
    plt.legend() # 描画

    plt.show() # 描画

def calc_hamming(hmm: HMM) -> int:
    return np.sum(np.absolute(hmm.x - hmm.xmap))

def make_theta_list():
    theta_list = []
    theta = theta_first + theta_step
    while theta < theta_last:
        theta_list.append(theta)
        theta += theta_step
        theta = round(theta,2)
    
    return theta_list

def task(sigma, theta):
    hmm = [HMM(n,sigma,theta,theta) for i in range(hmm_num)]
    hamming_distanse = [0 for i in range(hmm_num)]

    #データモデル固定
    hmm[0].generate_x()
    for i in range(1,hmm_num):
        hmm[i].x = hmm[0].x

    for i in range(hmm_num):
        hmm[i].generate_y()
        hmm[i].compute_xmap()
        hamming_distanse[i] = calc_hamming(hmm[i])

    hd_mean = np.mean(hamming_distanse)
    hd_std = np.std(hamming_distanse)

    return hd_mean, hd_std


if __name__ == '__main__':
    hmm07()
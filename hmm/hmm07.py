import math
import random
import numpy
import matplotlib.pyplot as plt

n = 200
hmm_num = 1000
theta_first = 0.7
theta_last = 1.0
theta_step = 0.05

def make_matrix(a, b, fill=0.0):
    m = []
    for i in range(a):
        m.append([fill]*b)
    return m

class HMM:
    def __init__(self, n, sigma, p00, p11):
        self.n = n
        self.sigma = sigma
        self.S = make_matrix(self.n, 2)     #次のをみて、Xiが取るべき値
        self.C = make_matrix(self.n, 2)     #もっともらしさ

        self.p00 = p00
        self.p01 = 1.00 - p00
        self.p10 = 1.00 - p11
        self.p11 = p11

        self.x = [0]*self.n
        self.xmap = [0]*self.n
        self.y = [0.0]*self.n
        
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
        
        if self.C[self.n-1][0] > self.C[self.n-1][1]:
            self.S[self.n-1][0] = 0
            self.S[self.n-1][1] = 0
        else:
            self.S[self.n-1][0] = 1
            self.S[self.n-1][1] = 1

        for i in range(0,self.n):
            if self.C[i][0] > self.C[i][1]:
                self.xmap[i] = self.S[i][0]
            else:
                self.xmap[i] = self.S[i][1]


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
    plt.xticks(numpy.arange(min(theta_list), max(theta_list) + 0.01, theta_step))
    plt.legend() # 描画

    plt.show() # 描画

def calc_hamming(hmm):
    hamming_distanse = 0
    for i in range(hmm.n):
        hamming_distanse += abs(hmm.x[i] - hmm.xmap[i])
    return hamming_distanse

def make_theta_list():
    theta_list = []
    theta = theta_first + theta_step
    while theta < theta_last:
        theta_list.append(theta)
        theta += theta_step
        theta = round(theta,2)
    
    return theta_list

def task(sigma, theta):
    hmm = [HMM(n,sigma,theta,theta)] * hmm_num
    hamming_distanse = [0] * hmm_num

    #データモデル固定
    hmm[0].generate_x()
    for i in range(1,hmm_num):
        hmm[i].x = hmm[0].x

    for i in range(hmm_num):
        hmm[i].generate_y()
        hmm[i].compute_xmap()
        hamming_distanse[i] = calc_hamming(hmm[i])

    hd_mean = numpy.mean(hamming_distanse)
    hd_std = numpy.std(hamming_distanse)

    return hd_mean, hd_std


if __name__ == '__main__':
    hmm07()
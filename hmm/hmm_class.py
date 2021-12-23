# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt

def make_matrix(a, b, fill=0.0): #  NumPy を使って高速に処理する方法がある
    m = []
    for i in range(a):
        m.append([fill]*b)
    return m

class HMM:
    def __init__(self, n, sigma):
        self.n = n
        self.sigma = sigma
        self.S = make_matrix(self.n, 2)     # 次のをみて、Xiが取るべき値
        self.C = make_matrix(self.n, 2)     # もっともらしさ

        self.x = [0]*self.n
        self.xmap = [0]*self.n
        self.y = [0.0]*self.n
        
        self.T = 2*pow(sigma,2)

        self.log_p00 = math.log(0.99)
        self.log_p01 = math.log(0.01)
        self.log_p10 = math.log(0.03)
        self.log_p11 = math.log(0.97)


    def generate_x(self):
        if (random.random() < 0.5):
            self.x[0] = 0
        else:
            self.x[0] = 1

        for i in range(1,self.n):
            r = random.random()
            if ( self.x[i-1] == 0 ):
                if ( r < 0.99 ):
                    self.x[i] = 0
                else:
                    self.x[i] = 1
            else:
                if ( r < 0.97 ):
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
                self.C[i][0] += self.C[i-1][0] + self.log_p00
            else:
                self.S[i-1][0] = 1
                self.C[i][0] += self.C[i-1][1] + self.log_p10

            if t01 > t11:
                self.S[i-1][1] = 0
                self.C[i][1] += self.C[i-1][0] + self.log_p01
            else:
                self.S[i-1][1] = 1
                self.C[i][1] += self.C[i-1][1] + self.log_p11
        
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

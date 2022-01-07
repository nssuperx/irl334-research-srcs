import math
import random
import matplotlib.pyplot as plt

from hmm_class import HMM

def hmm01():
    n = 200
    sigma = 0.7
    hmm = [HMM(n,sigma)] * 10

    t = range(n)
    fig=plt.figure(0)

    for i in range(10):
        hmm[i].generate_x()
        for j in range(n):
            hmm[i].x[j] += i * 3
        plt.plot(t,hmm[i].x)

    plt.title('X example') 
    plt.xlabel('x') # X 軸
    plt.ylabel('y') # Y 軸

    xlabel = ['x'] * 10
    for i in range(10):
        xlabel[i] = 'x' + str(i)
    plt.legend(xlabel) # 描画

    plt.show() # 描画
    fig.savefig('fig1.pdf')

if __name__ == '__main__':
    hmm01()
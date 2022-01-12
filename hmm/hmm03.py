import math
import random

from hmm_class import HMM

def make_matrix(a, b, fill=0.0):
    m = []
    for i in range(a):
        m.append([fill]*b)
    return m


def test20():
    n = 200
    m = 20 # m test cases
    sigma = 0.7
    hmm = HMM(n, sigma)
    z = make_matrix(m*2, n) 

    i = 0
    for line in open("test_cases.txt", "r"):
        if line[0] == "#":
            continue
        data = line.split() # 文字列を空白文字を区切りに分割

        for a in range(0,m):
#            print(a*2, i, len(data))
            z[a*2][i] = float(data[a*2])
            z[a*2+1][i] = float(data[a*2+1])
            
        i=i+1
    # データ読み込み終了    
   
    for a in range(0,m):
        for i in range(0,n):
            hmm.y[i] = z[a*2][i]
        hmm.compute_xmap()
        num_pass = 0
        for i in range(0,n):
#            print((hmm.xmap[i] + 2), int(z[a*2+1][i]))
            if (hmm.xmap[i] + 2 == z[a*2+1][i]):
                num_pass=num_pass+1
        if ( num_pass == n):
            print(a,":" , "Passed !!!")
        else:
            print(a,":" , "not passed")

if __name__ == '__main__':
    test20()
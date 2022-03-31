import math
import numpy as np

def generate_x(size, p00, p11):
    x = np.zeros(size, dtype=np.int8)

    for i in range(1,size):
        r = np.random.random()
        if ( x[i-1] == 0 ):
            x[i] = 0 if r < p00 else 1
        else:
            x[i] = 1 if r < p11 else 0
    return x

def compute_xmap(size, sigma, y, p00, p01, p10, p11):
    log_p00 = math.log(p00)
    log_p01 = math.log(p01)
    log_p10 = math.log(p10)
    log_p11 = math.log(p11)

    log_sigma = math.log(sigma)

    dp = np.zeros((size, 2), dtype=np.int8)        # 次のをみて、Xiが取るべき値
    plausible = np.zeros((size, 2), dtype=np.float32)     # もっともらしさ
    xmap = np.zeros(size, dtype=np.int8)

    plausible[0][0] = -pow(y[0] - 0.0, 2.0) / (2.0*pow(sigma, 2.0)) - log_sigma
    plausible[0][1] = -pow(y[0] - 1.0, 2.0) / (2.0*pow(sigma, 2.0)) - log_sigma

    for i in range(1,size):
        plausible[i][0] = -pow((y[i] - 0.0), 2.0)/ (2.0*pow(sigma, 2.0)) - log_sigma
        if(plausible[i-1][0] + log_p00 > plausible[i-1][1] + log_p10):
            dp[i-1][0] = 0
            plausible[i][0] += plausible[i-1][0] + log_p00
        else:
            dp[i-1][0] = 1
            plausible[i][0] += plausible[i-1][1] + log_p10

        plausible[i][1] = -pow((y[i] - 1.0), 2.0)/ (2.0*pow(sigma, 2.0)) - log_sigma
        if(plausible[i-1][0] + log_p01 > plausible[i-1][1] + log_p11):
            dp[i-1][1] = 0
            plausible[i][1] += plausible[i-1][0] + log_p01
        else:
            dp[i-1][1] = 1
            plausible[i][1] += plausible[i-1][1] + log_p11
    
    xmap[size-1] = 0 if plausible[size-1][0] > plausible[size-1][1] else 1

    for i in range(2,size+1):
        xmap[size-i] = dp[size-i][xmap[size-i+1]]

    return xmap

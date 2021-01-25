import numpy as np

class NMF:
    W = None
    H = None
    loss_LOG = None

    epsilon = None

    def __init__(self, epsilon=1e-7, seed=0):
        self.epsilon = epsilon
        np.random.seed(seed)

    def calc(self, V, r, iteration=500):
        """
        NMF計算

        Parameters
        ----------
        V: numpy.adarray
            オリジナルのデータ
        r: int
            基底数
        iteration: int
            繰り返し回数
        """
        if (r <= 0):
            print("NMF calc error!!")
            print("r must be greater than zero.")
            exit()
        
        self.W = np.random.rand(V.shape[0], r)
        self.H = np.random.rand(r, V.shape[1])
        self.loss_LOG = []
        for i in range(iteration):
            self.update(V)
            loss = self.kl_divergence(V)
            self.loss_LOG.append(loss)

    def update(self, V):
        """
        NMF更新
        Lee & Seung アルゴリズム

        Parameters
        ----------
        V: numpy.adarray
            オリジナルのデータ
        """
        WH = np.dot(self.W, self.H) + self.epsilon
        self.W = self.W * np.dot(V / WH, self.H.T)

        W_tmp = np.sum(self.W, axis=0)
        self.W = self.W / np.tile(W_tmp, (V.shape[0], 1))

        WH = np.dot(self.W, self.H) + self.epsilon
        self.H = self.H * np.dot(self.W.T, (V/WH))

    def kl_divergence(self, V):
        WH = np.dot(self.W, self.H) + self.epsilon
        F = np.sum(np.multiply(V, np.log(WH)) - WH)
        return F
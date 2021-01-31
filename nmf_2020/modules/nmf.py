import numpy as np

class NMF:
    W = None
    H = None
    loss_LOG = None
    # probdist_V = None

    epsilon = None

    def __init__(self, epsilon=1e-7, seed=0):
        self.epsilon = epsilon
        np.random.seed(seed)

    def calc(self, V, r, iteration=500):
        """
        NMF計算

        Parameters
        ----------
        V: numpy.ndarray
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
        
        # self.probdist_V = self.prob_dist(V)
        self.W = np.random.rand(V.shape[0], r)
        self.H = np.random.rand(r, V.shape[1])
        self.loss_LOG = []
        for i in range(iteration):
            self.update(V)
            loss = self.kl_divergence(V)
            # loss = self.frobenius_norm(V)
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

        self.W = self.W / np.tile(np.sum(self.W, axis=0), (self.W.shape[0], 1))

        WH = np.dot(self.W, self.H) + self.epsilon
        self.H = self.H * np.dot(self.W.T, (V / WH))

    def kl_divergence(self, V):
        WH = np.dot(self.W, self.H) + self.epsilon
        F = np.sum(np.multiply(V, np.log(WH)) - WH)
        return F

    def frobenius_norm(self, V):
        WH = np.dot(self.W, self.H) + self.epsilon
        F = np.linalg.norm(V - WH)
        return F

    def prob_dist(self, V):
        return V / np.tile(np.sum(V, axis=0), (V.shape[0], 1))

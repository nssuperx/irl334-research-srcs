import numpy as np

class NMF:
    W = None
    H = None
    loss_LOG = None
    # probdist_V = None

    epsilon = None

    def __init__(self, epsilon=1e-7, seed=None):
        self.epsilon = epsilon
        np.random.seed(seed)

    def calc(self, V, r, iteration=500, save=False, use_cache=False):
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
        if use_cache:
            self.load_data()
            return

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

        if save:
            self.save_data()

    def update(self, V):
        """
        NMF更新
        Lee & Seung アルゴリズム

        Parameters
        ----------
        V: numpy.adarray
            オリジナルのデータ
        """
        self.W = self.W * np.dot(V / (np.dot(self.W, self.H) + self.epsilon), self.H.T)
        self.W = self.W / np.tile(np.sum(self.W, axis=0), (self.W.shape[0], 1))
        self.H = self.H * np.dot(self.W.T, (V / (np.dot(self.W, self.H) + self.epsilon)))

    def kl_divergence(self, V):
        """
        KL-divergenceを計算する．

        Parameters
        ----------
        V: numpy.adarray
            オリジナルのデータ

        Returns
        ----------
        F: float
            KL-divergence
        """
        WH = np.dot(self.W, self.H) + self.epsilon
        F = np.sum(np.multiply(V, np.log(WH)) - WH)
        # F = np.sum(np.multiply(V, np.log(WH)) - WH) / V.shape[1]
        # F = np.sum(np.multiply(V, np.log(WH)) - WH) / (V.shape[0] * V.shape[1])
        return F

    def frobenius_norm(self, V):
        """
        frobeniusノルムを計算する．

        Parameters
        ----------
        V: numpy.adarray
            オリジナルのデータ

        Returns
        ----------
        F: float
            frobeniusノルム
        """
        WH = np.dot(self.W, self.H) + self.epsilon
        F = np.linalg.norm(V - WH)
        return F

    def prob_dist(self, V):
        """
        行列をaxis=0の方向で確率分布の行列にする．

        Parameters
        ----------
        V: numpy.adarray
            行列

        Returns
        ----------
        numpy.adarray
            確率分布になった行列
        """
        return V / np.tile(np.sum(V, axis=0), (V.shape[0], 1))

    def save_data(self, filename=None):
        """
        計算後のW,H,lossのログを保存する．
        連続して実験するときに使う．

        Parameters
        ----------
        filename: str
            ファイル名文字列

        """
        if filename is None:
            fn = 'tmp'
        else:
            fn = str(filename)

        np.savez(fn, W=self.W, H=self.H, loss=np.array(self.loss_LOG))

    def load_data(self, filename=None):
        """
        計算後のW,H,lossのログを読み込む．
        連続して実験するときに使う．

        Parameters
        ----------
        filename: str
            ファイル名文字列

        """
        if filename is None:
            fn = 'tmp'
        else:
            fn = str(filename)
        
        load_array = np.load(fn + '.npz')
        self.W = load_array['W']
        self.H = load_array['H']
        self.loss_LOG = load_array['loss'].tolist()

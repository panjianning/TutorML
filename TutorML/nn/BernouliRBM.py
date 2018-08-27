import numpy as np
from scipy.special import expit

def sigmoid(x):
    return expit(x)

def sample_beurnouli(prob):
    sh = prob.shape
    return (np.random.rand(*sh) < prob).astype(int)

class BernouliRBM(object):
    ''' p(v,h) = 1/Z*exp(-E(v,h))
        E(v,h) = -(v^TWh+v^Tb+h^Tc)
        W shape: (n_visibles, n_hiddens)
    '''
    def __init__(self, n_visibles, n_hiddens):
        self.n_visibles = n_visibles
        self.n_hiddens = n_hiddens

    def _init_parameters(self):
        self.W = np.random.randn(self.n_visibles, self.n_hiddens)
        self.b = np.zeros((self.n_visibles,1))
        self.c = np.zeros((self.n_hiddens,1))

    def fit(self,X, n_epochs=10, batch_size=10, learning_rate=0.1):
        self._init_parameters()
        n_samples = X.shape[0]
        for epoch in range(1, n_epochs+1):
            i = 0
            while i < n_samples:
                # empirical
                V = X[i:i+batch_size]
                H = sigmoid(np.matmul(V, self.W) + self.c.T)
                # CD-1
                sample_H  = sample_beurnouli(H)
                V1 = sigmoid(np.matmul(sample_H, self.W.T) + self.b.T)
                sample_V1 = sample_beurnouli(V1)
                H1 = sigmoid(np.matmul(V1, self.W) + self.c.T)
                # take expectation
                v_mean = np.mean(V, axis=0).reshape(-1,1)
                h_mean = np.mean(H, axis=0).reshape(-1,1)
                v1_mean = np.mean(sample_V1, axis=0).reshape(-1,1)
                h1_mean = np.mean(H1, axis=0).reshape(-1,1)
                # gradient update
                dw = np.matmul(v_mean, h_mean.T) - np.matmul(v1_mean, h1_mean.T)
                # self.W += learning_rate/n_samples * np.matmul(V.T, H)
                self.W += learning_rate * dw
                self.b += learning_rate * (v_mean - v1_mean)
                self.c += learning_rate * (h_mean - h1_mean)
                i += batch_size
    def transform(self, X):
        return sigmoid(np.matmul(X, self.W))
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

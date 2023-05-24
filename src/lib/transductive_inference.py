from .graphs_general.adjacency_list import AdjacencyList
from .graphs_general.adjacency_matrix import AdjacencyMatrix

import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial.distance import cdist

import numpy as np
import math

class TransductiveInference:
    
    def __init__(self, pd_data: pd.Series,
                 graph_type: (AdjacencyList | AdjacencyMatrix) = AdjacencyMatrix) -> None:
        self.__adapter = _PandasAdapter(pd_data)
        #self.__graph = (self.__adapter.get_adjacency_matrix()
        #                if graph_type == AdjacencyMatrix else self.__adapter.get_adjacency_list())
        self.affinity_matrix = self.create_affinity_matrix()

        self.s = self.create_S()
        self.alpha = 0.95

    
    def create_affinity_matrix(self):
        dm = cdist(self.__adapter.x, self.__adapter.x, 'euclidean')
        
        rbf = lambda x, std: math.exp((-x) / (2 * (math.pow(std, 2))))
        vfunc = np.vectorize(rbf)
        
        w = vfunc(dm, self.__adapter.std)
        np.fill_diagonal(w, 0)
        
        return w

    def create_S(self):
        d = np.sum(self.affinity_matrix, axis=1)
        D = np.sqrt(d * d[:, np.newaxis])

        return np.divide(self.affinity_matrix, D, where= D != 0)

    def y_input(self):
        res = np.ndarray(shape=(self.__adapter.x.shape[0], self.__adapter.x.shape[0]))
        
        c = 0
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i][j] = self.__adapter.y[c] if c < self.__adapter.train_size else 0.0
                c += 1
        return res

    def fit(self):
        y_input = self.y_input()
        F = np.dot(self.s, y_input) * self.alpha + (1 - self.alpha) * y_input

        Y_result = np.zeros_like(F)
        Y_result[np.arange(len(F)), np.argmax(1)] = 1

        print("Y_result", Y_result[0:])
        Y_v = [1 if x == 0 else 0 for x in Y_result[0]]

        color = ['red' if l == 0 else 'blue' for l in Y_v]
        
        total_y = np.ndarray(shape=(len(self.__adapter.x),))
        
        for i in range(len(total_y)):
            total_y[i] = self.__adapter.y[i] if i < len(self.__adapter.x) else 0.0

        print(self.__adapter.x)
        plt.scatter(self.__adapter.x[0:,0], self.__adapter.x[0:,1], color=color)
        #plt.savefig("iter_1.pdf", format='pdf')
        plt.show()

class _PandasAdapter:
    """'Private' adapter to get a pandas DataFrame class with two features X and Y
    (or pandas Series with only X because Y is a simple label L = {1, ... l} l < len(X), acording
    to the article) and write it in a graph.
    """

    def __init__(self, data: pd.Series, train_size: float = 0.1, random_state=None) -> None:
        if 0.0 >= train_size >= 1.0:
            raise ValueError("Train size must in the open interval (0, 1).")
        
        self.data = data.copy()

        self.x = self.data.sample(frac=train_size, random_state=random_state) # values itself

        self.train_size = int(len(self.data) * train_size)

        self.x = self.x.to_numpy()
        self.y = np.array([v for v in range(1, self.train_size + 1)])

        self.std = self.data.std().mean()
        self.mean = self.data.mean()

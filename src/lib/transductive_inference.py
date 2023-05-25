from .graphs_general.adjacency_list import AdjacencyList
from .graphs_general.adjacency_matrix import AdjacencyMatrix

import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial.distance import cdist

import numpy as np
import math

class TransductiveInference:
    
    def __init__(self, pd_data: pd.Series,
                 graph_type: (AdjacencyList | AdjacencyMatrix) = AdjacencyMatrix,
                 n_iter: int = 1) -> None:
        self.__adapter = _PandasAdapter(pd_data)
        #self.__graph = (self.__adapter.get_adjacency_matrix()
        #                if graph_type == AdjacencyMatrix else self.__adapter.get_adjacency_list())
        self.affinity_matrix = self.create_affinity_matrix()

        self.s = self.create_S()
        self.alpha = 0.99
        self.n_iter = n_iter

    
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
        return np.concatenate(((self.__adapter.y[:, None] == np.arange(1, self.__adapter.n_labeled + 1)).astype(float),
                               np.zeros((self.__adapter.n_not_labeled, self.__adapter.n_labeled))))

    def fit(self):
        y_input = self.y_input()
        
        F = np.dot(self.s, y_input) * self.alpha + (1 - self.alpha) * y_input
        for _ in range(self.n_iter):
            F = np.dot(self.s, F) * self.alpha + (1 - self.alpha) * y_input

        print("Final F:", F)

        Y_result = np.zeros_like(F)
        # Y_result[np.arange(len(F)), np.argmax(1)] = 1

        for i in range(len(F)):
            dim = len(F[0])
            max = 0
            
            for j in range(dim):
                if F[i][j] > F[i][max]:
                    max = j
            Y_result[i][0] = max

        print()
        print("Classification results:", Y_result[0:,0])
        
        # Y_v = [1 if x == 0 else 0 for x in Y_result[0:,0]]
        Y_v = [x for x in Y_result[0:,0]]

        color = []
        for l in Y_v:
            if l == 0:
                color.append('red')
            elif l == 1:
                color.append('blue')
            elif l == 2:
                color.append('orange')
            elif l == 3:
                color.append('purple')
            else:
                color.append('black')
        
        plt.scatter(self.__adapter.x[0:,0], self.__adapter.x[0:,1], color=color)
        #plt.savefig("iter_1.pdf", format='pdf')
        plt.show()

class _PandasAdapter:
    def __init__(self, data: pd.Series, n_labeled_frac: float = (1/125)) -> None:
        if 0.0 >= n_labeled_frac >= 1.0:
            raise ValueError("Train size must in the open interval (0, 1).")
        
        self.data = data.copy()

        # self.x = # self.data.sample(frac=n_labeled_frac, random_state=random_state) # values itself

        self.n_labeled = int(len(self.data) * n_labeled_frac)
        self.n_not_labeled = len(self.data) - self.n_labeled
        
        self.x = self.data.to_numpy()
        self.y = np.array([v for v in range(1, self.n_labeled + 1)])

        self.std = self.data.std().mean()
        self.mean = self.data.mean()

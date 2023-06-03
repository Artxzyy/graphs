from .graphs_general.adjacency_list import AdjacencyList
from .graphs_general.adjacency_matrix import AdjacencyMatrix

import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial.distance import cdist

import numpy as np
import math

class TransductiveInference:

    __color_labels = {
        0: "red", 1: "blue",
        2: "green", 3: "orange",
        4: "purple", 5: "gray",
        6: "black", 7: "cyan",
        8: "magenta", 9: "yellow",
        10: "magenta", 11: "yellow",
        12: "magenta", 13: "yellow",
        14: "magenta", 15: "yellow",
        16: "magenta", 17: "yellow"
    }

    def __init__(self, pd_data, y, y_name: str = "", n_iter: int = 1000, alpha: float = 0.95) -> None:
        self.__adapter = _PandasAdapter(pd_data, y, y_name)

        self.__affinity_matrix = self.__create_affinity_matrix()
        
        self.__s = self.__create_S()
        self.__alpha = alpha
        self.__n_iter = n_iter

        self.__result = None

    
    def __create_affinity_matrix(self):
        dm = cdist(self.__adapter.x, self.__adapter.x, 'euclidean')
        
        rbf = lambda x, std: math.exp((-x) / (2 * (math.pow(std, 2))))
        vfunc = np.vectorize(rbf)
        
        w = vfunc(dm, self.__adapter.std)
        np.fill_diagonal(w, 0)
        
        return w

    def __create_S(self):
        d = np.sum(self.__affinity_matrix, axis=1)
        D = np.sqrt(d * d[:, np.newaxis])

        return np.divide(self.__affinity_matrix, D, where= D != 0)

    def __y_input(self):
        #Y_input = np.concatenate(((Y[:n_labeled,None] == np.arange(2)).astype(float), np.zeros((n-n_labeled,2))))
        res = np.concatenate(((self.__adapter.y[:self.__adapter.n_labeled,None] == np.arange(self.__adapter.unique_labels)).astype(float),
                             np.zeros((self.__adapter.n_not_labeled, self.__adapter.unique_labels))))
        return res
        #return np.concatenate(((self.__adapter.y[:self.__adapter.n_labeled, None] == np.arange(self.__adapter.unique_labels)).astype(float),
        #                       np.zeros((self.__adapter.n_not_labeled, self.__adapter.unique_labels))))

    def fit_predict(self):
        y_input = self.__y_input()

        F = np.dot(self.__s, y_input) * self.__alpha + (1 - self.__alpha) * y_input # 1st iter
        for _ in range(self.__n_iter):
            F = np.dot(self.__s, F) * self.__alpha + (1 - self.__alpha) * y_input

        Y_result = np.zeros_like(F)

        for i in range(len(F)):
            max = 0
            dim = len(F[0])
            
            for j in range(dim):
                if F[i][j] > F[i][max]:
                    max = j
            Y_result[i][0] = max

        Y_v = [x for x in Y_result[0:,0]]

        color = []
        for l in Y_v:
            color.append(TransductiveInference.__color_labels[l])

        self.__result = Y_v.copy()
        self.__colors = color

        return self.__result
    
    def plot(self, title: str = "Title", xlabel: str = "X Label", ylabel: str = "Y Label",
             save: bool = False, save_name_pdf: str = "plot.pdf"):
        if self.__result is None:
            raise ValueError("The model was not trained yet. Call fit_predict() to train the model first.")
        
        plt.title(title)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.scatter(self.__adapter.x[0:,0], self.__adapter.x[0:,1], color=self.__colors)
        
        if save:
            plt.savefig(save_name_pdf, format='pdf')
        
        plt.show()

class _PandasAdapter:
    def __init__(self, data, y, y_name: str, n_labeled_frac: float = .25) -> None:
        if 0.0 >= n_labeled_frac >= 1.0:
            raise ValueError("Train size must in the open interval (0, 1).")
        
        self.data = data.copy()

        self.n_labeled = int(len(self.data) * n_labeled_frac)
        self.n_not_labeled = len(self.data) - self.n_labeled

        self.x = None
        self.y = None
        self.unique_labels = None

        if type(data) == type(y) == np.ndarray:
            self.x = self.data.copy()
            self.y = y.copy()
            self.unique_labels = len(np.unique(self.y))
        elif type(data) == type(y) == pd.DataFrame or type(data) == pd.DataFrame and type(y) == pd.Series:
            self.x = self.data.drop(y_name, axis=1).to_numpy()
            self.y = y.copy().to_numpy()
            self.unique_labels = len(self.data[y_name].unique())
        else:
            raise TypeError("data and y types must be either both numpy.ndarray or data=pandas.DataFrame and y=pandas.DataFrame or pandas.Series")

        std = self.x.std().mean()
        self.std = .2

        #self.mean = self.x.mean()

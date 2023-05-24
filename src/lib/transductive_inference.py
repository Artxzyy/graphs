from .graphs_general.adjacency_list import AdjacencyList
from .graphs_general.adjacency_matrix import AdjacencyMatrix

import pandas as pd
import math

class TransductiveInference:
    """
    Semi-supervised machine-learning model proposed at
    https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf

    'Learning with Local and Global Consistency'
    
    """
    
    def __init__(self, pd_data: pd.Series,
                 graph_type: (AdjacencyList | AdjacencyMatrix) = AdjacencyMatrix) -> None:
        self.__adapter = _PandasAdapter(pd_data)
        self.__graph = (self.__adapter.get_adjacency_matrix()
                        if graph_type == AdjacencyMatrix else self.__adapter.get_adjacency_list())
        self.__affinity_matrix = self.create_affinity_matrix()

        self.__mean = self.__adapter.get_mean()
        self.__std_deviation = self.__adapter.get_std_deviation()

    def affinity_mapping(self, xi, xj) -> float:
        """Still yet to get mean and std_deviation for xi and xj instead"""
        return (1 / (((2 * math.pi * (self.__adapter.get_std_deviation() ** (2)))) ** (1 / 2)))
    
    def std(self):
        return self.__adapter.get_std_deviation()
    
    def create_affinity_matrix(self) -> list[list]:
        """Use adapter's x property to create matrix"""
        res = []
        
        x = self.__adapter.get_x()
    

        # change the following to applying the affinity_mapping
        # as a numpy array an then fill diagonal with zeros
        size = len(x)
        for i in range(size):
            res.append([]) # i-th row
            for j in range(size):
                res[i].append(self.affinity_mapping(x[i], x[j]) if i != j else 0.0) # j-th column
        return res

    def create_diagonal_matrix(self) -> list[list]:
        pass


class _PandasAdapter:
    """'Private' adapter to get a pandas DataFrame class with two features X and Y
    (or pandas Series with only X because Y is a simple label L = {1, ... l} l < len(X), acording
    to the article) and write it in a graph.
    """

    def __init__(self, data: pd.Series, train_size: float = 0.8) -> None:
        if 0.0 >= train_size >= 1.0:
            raise ValueError("Train size must in the open interval (0, 1).")
        
        self.__data = data.copy()

        self.__x = tuple(self.__data.to_numpy()) # values itself

        self.__train_size = int(len(self.__x) * train_size)
        self.__y = tuple([i for i in range(1, self.__train_size + 1)]) # labels: tuple with (label=Nullable, value=NonNullable)

    def get_std_deviation(self) -> float:
        return self.__data.std()
    
    def get_mean(self) -> float:
        return self.__data.mean()
    
    def get_adjacency_matrix(self) -> AdjacencyMatrix:
        """Supposing only vertex weight."""
        return AdjacencyMatrix(n=len(self.__x), weights=self.__x)
    
    def get_adjacency_list(self) -> AdjacencyList:
        """Supposing only vertex weight."""
        return AdjacencyList(n=len(self.__x), weights=self.__x)
    
    def get_x(self) -> tuple[int]:
        return self.__x
    
    def get_y(self) -> tuple[int]:
        return self.__y



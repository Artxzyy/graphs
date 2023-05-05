from general.adjacency_list import AdjacencyList
from general.adjacency_matrix import AdjacencyMatrix

import pandas as pd

class TransductiveInference:
    """
    Semi-supervised machine-learning model proposed at
    https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf
    """
    
    def __init__(self) -> None:
        pass

    def get_pandas_data(self, pd_data, graph_type: (AdjacencyList | AdjacencyMatrix) = AdjacencyMatrix):
        return (_PandasAdapter(pd_data, "x_name", "y").get_adjacency_matrix()
                if graph_type == AdjacencyMatrix else _PandasAdapter(pd_data, "x_name", "y").get_adjacency_list())

class _PandasAdapter:
    """'Private' adapter to get a pandas DataFrame class with two features X and Y
    (or pandas Series with only X because Y is a simple label L = {1, ... l} l < len(X), acording
    to the article) and write it in a graph."""

    def __init__(self, pandas_data: pd.DataFrame) -> None:
        self.__data = pandas_data.copy()

        self.__x_name = None
        self.__x = None # values itself
        
        self.__y = None # labels: tuple with (label=Nullable, value=NonNullable)
    
    def get_adjacency_matrix(self):
        return None
    
    def get_adjacency_list(self):
        return None



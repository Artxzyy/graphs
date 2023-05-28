class _Graph:
    def __init__(self, n: int = 5, directed: bool = False,
                 v_labeled: bool = False, v_weighted: bool = False,
                 e_labeled: bool = False, e_weighted: bool = False) -> None:
        """
        AdjacencyList constructor

        Parameters

        n: int = 5
            starting number of vertexes
        v_labeled: bool = False
            boolean that defines if the vertices should have labels
        v_weighted: bool = False
            boolean that defines if the vertices should have weights
        e_labeled: bool = False
            boolean that defines if the edges should have labels
        e_weighted: bool = False
            boolean that defines if the edges should have weights
        """
        
        self._n = n
        self._edges_counter = 0
        
        self._directed = directed

        self._v_labeled = v_labeled
        self._v_weighted = v_weighted

        if not self._v_labeled:
            self._vertex_iterable = 1

        self._e_labeled = e_labeled
        self._e_weighted = e_weighted

        if not self._e_labeled:
            self._edge_iterable = 1

    def info(self) -> dict:
        """Returns a dictionary with how this graph was built and how which parameters were defined."""
        return {
            "directed": self._directed,
            "v_labeled": self._v_labeled,
            "v_weighted": self._v_weighted,
            "e_labeled": self._e_labeled,
            "e_weighted": self._e_weighted
        }

    def size(self) -> int:
        """Returns the amount of vertexes in the graph."""
        return self._n
    
    def edges(self) -> int:
        """Returns the amount of edges in the graph."""
        return self._edges_counter
    
    def complete(self) -> bool:
        """Tests if amount of edges is equal to ((n * (n - 1)) / 2). If equal, and supposing the graph is simple,
        returns True; otherwise, False."""
        return (self._edges_counter == (self._n * (self._n - 1)) / 2)

    def v_labeled(self) -> bool:
        """Returns if the graph has labels for vertexes."""
        return self._v_labeled
    
    def v_weighted(self) -> bool:
        """Returns if the graph has weights for vertexes."""
        return self._v_weighted
    
    def e_labeled(self) -> bool:
        """Returns if the graph has labels for edges."""
        return self._e_labeled
    
    def e_weighted(self) -> bool:
        """Returns if the graph has weights for edges."""
        return self._e_weighted
    
    def empty(self) -> bool:
        """Returns if the amount of vertexes in the graph equals 0."""
        return self._n == 0
from vertex.vertex import Vertex
from edge.edge import Edge

class AdjacencyList:

    def __init__(self, n: int = 5, labels: tuple[(str | int)] = None, weights: tuple[float] = None, directed: bool = False) -> None:
        if n < 0:
            raise ValueError(f"N must be greater than or equal to zero, but received {n}.")
        

        self.__n = n
        self.__edges_counter = 0
        self.__directed = directed
        self.vertexes = {} # {VertexOne: [EdgesOne], VertexTwo: [EdgesTwo]}

        self.__labeled = False if labels is None else True
        self.__weighted = False if weights is None else True

        if not self.__labeled:
            self.__iterable = 1

        tmp_vertexes = self.create_vertex(n, labels, weights)

        for v in tmp_vertexes:
            self.vertexes[v] = []

    def add_vertex(self, vertex: (Vertex | tuple[Vertex]),
                   label: (str | int | tuple[str] | tuple[int] | None) = None,
                   weight: (float | tuple[float] | None) = None) -> None:
        """It is supposed that if vertex is a tuple o vertexes, label and weight area also
        tuples with all the same length. If vertex is not a tuple, it is just a Vertex object,
        label is just a string or int or None and weight is just a float or None"""

        if self.__labeled and self.__weighted:
            if not (len(vertex) == len(label) == len(weight)):
                raise IndexError(f"The length for labels ({len(label)}) and weights ({len(weight)}) \
                                 must be equal to n ({len(vertex)}).")
        elif self.__labeled:
            if len(label) != len(vertex):
                raise IndexError(f"The length for labels ({len(label)}) must be equal to n ({len(vertex)}).")
        elif self.__weighted:
            if len(weight) != len(vertex):
                raise IndexError(f"The length for weights ({len(weight)}) must be equal to n ({len(vertex)}).")
                
        if type(vertex) != tuple:
            v = self.create_vertex(1, ((label,) if self.__labeled else None),
                                   ((weight,) if self.__weighted else None))
            v = list(v.keys())[0]
            self.vertexes[v] = []
            self.__n += 1
        else:
            v = self.create_vertex(len(vertex), (label if self.__labeled else None),
                                   (weight if self.__weighted else None))
            for i in v:
                self.vertexes[i] = []
                self.__n += 1


    def create_vertex(self, n: int, labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None) -> dict:
        if labels is not None and len(labels) != n:
            raise IndexError(f"The length for labels ({len(labels)}) must be equal to n ({n}).")
        if weights is not None and len(weights) != n:
            raise IndexError(f"The length for weights ({len(weights)}) must be equal to n ({n}).")

        n_labels = tuple([i for i in range(self.__iterable, self.__iterable + n)]) if labels is None else labels
        n_weights = ((None,) * n) if weights is None else weights

        if labels is None:
            self.__iterable += n
        
        res = {}
        for i in range(n):
            res[Vertex(n_labels[i], n_weights[i])] = []

        return res

    def add_edge(self, v: (Vertex | str | int), w: (Vertex | str | int),
                 label: (str | int | None) = None, weight: (float | None) = None) -> None:
        # supposing v and w are Vertexes

        # search for v
        # search for w
        # if both exist:
            # create edge from v to w with label and weight
            # if graph is not directed
                # also creates edge from w to v with label and weight
                
        if self.search_vertex(v) == -1:
            raise ValueError("Vertex v was not found in the Adjacency List.")
        if self.search_vertex(w) == -1:
            raise ValueError("Vertex w was not found in the Adjacency List.")
        
        self.vertexes[v].append(Edge(w, label, weight))

        if not self.__directed:
            self.vertexes[w].append(Edge(v, label, weight))

        self.__edges_counter += 1


    def search_vertex(self, v: Vertex) -> int:
        res = -1
        for i, j in enumerate(self.vertexes):
            if j == v:
                res = i
                break
        return res
    
    def print(self) -> None:
        """v -> w1, w2, w3 ..."""

        print("Adjacency list vertexes\n")
        
        for v in self.vertexes:
            print(v.to_string(), " -> ", sep="", end="")
        
            if len(self.vertexes[v]) > 0:
                print(self.vertexes[v][0].to.get_label(), sep="", end="")
        
                for e in self.vertexes[v][1:]:
                    print(", ", e.to.get_label(), sep="", end="")
            else:
                print("None", end="")
            print()
    
    def empty(self) -> bool:
        return self.__n == 0

al = AdjacencyList(n=5, weights=(1, 5, 2.1, 10, 15.5))

vs = [list(al.vertexes.keys())[0], list(al.vertexes.keys())[1]]

al.add_edge(vs[0], vs[1], "a1", "w1")

al.print()

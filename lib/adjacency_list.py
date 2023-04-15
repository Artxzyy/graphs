from vertex.vertex import Vertex

class AdjacencyList:

    def __init__(self, n: int = 5, labels: tuple[(str | int)] = None, weights: tuple[float] = None, directed: bool = False) -> None:
        if n < 0:
            raise ValueError(f"N must be greater than or equal to zero, but received {n}.")
        
        self.__n = n
        self.__directed = directed
        self.__vertexes = [] # [{vertex: [vertexes]}]

        self.__labeled = False if labels is None else True
        self.__weighted = False if weights is None else True

        if not self.__labeled:
            self.__iterable = 1

        tmp_vertexes = self.create_vertex(n, labels, weights)

        for v in tmp_vertexes:
            self.__vertexes.append({list(v.keys())[0]: []})



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
                                   ((weight,) if self.__weighted else None))[0]
            self.__vertexes.append({list(v.keys())[0]: []})
        else:
            v = self.create_vertex(len(vertex), (label if self.__labeled else None),
                                   (weight if self.__weighted else None))
            for i in v:
                self.__vertexes.append({list(i.keys())[0]: []})


    def create_vertex(self, n: int, labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None) -> tuple[Vertex]:
        if labels is not None and len(labels) != n:
            raise IndexError(f"The length for labels ({len(labels)}) must be equal to n ({n}).")
        if weights is not None and len(weights) != n:
            raise IndexError(f"The length for weights ({len(weights)}) must be equal to n ({n}).")

        n_labels = tuple([i for i in range(self.__iterable, self.__iterable + n)]) if labels is None else labels
        n_weights = ((None,) * n) if weights is None else weights

        if labels is None:
            self.__iterable += n

        return tuple([{Vertex(n_labels[i], n_weights[i]): []} for i in range(n)])

    def print(self) -> None:
        """v -> w1, w2, w3 ..."""
        print("Adjacency list vertexes\n")
        for v in self.__vertexes:
            print(list(v.keys())[0].to_string())


al = AdjacencyList(n=5, weights=(1, 5, 2.1, 10, 15.5))

al.print()

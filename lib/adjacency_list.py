from .vertex.vertex import Vertex
from .edge.edge import Edge

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

    def add_vertex(self, n: int,
                   label: (tuple[str] | tuple[int] | None) = None,
                   weight: (tuple[float] | None) = None) -> None:
        """It is supposed that if vertex is a tuple o vertexes, label and weight area also
        tuples with all the same length. If vertex is not a tuple, it is just a Vertex object,
        label is just a string or int or None and weight is just a float or None"""
        if n < 0:
            raise ValueError(f"N must be greater than or equal to zero, but received {n}.")
        
        if self.__labeled and self.__weighted:
            if not (n == len(label) == len(weight)):
                raise IndexError(f"The length for labels ({len(label)}) and weights ({len(weight)}) \
                                 must be equal to {n}.")
        elif self.__labeled:
            if len(label) != n:
                raise IndexError(f"The length for labels ({len(label)}) must be equal to {n}.")
        elif self.__weighted:
            if len(weight) != n:
                raise IndexError(f"The length for weights ({len(weight)}) must be equal to {n}.")

        v = self.create_vertex(n, label, weight)
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
        v_pos = self.search_vertex(v)
        w_pos = self.search_vertex(w)
        
        if v_pos == -1:
            raise ValueError("Vertex V was not found in the Adjacency List.")
        if w_pos == -1:
            raise ValueError("Vertex W was not found in the Adjacency List.")
    
        if type(v) == type(w) == Vertex:        
            self.vertexes[v].append(Edge(w, label, weight))
            if not self.__directed:
                self.vertexes[w].append(Edge(v, label, weight))
        elif type(v) == type(w) == str or type(v) == type(w) == int:
            keys = list(self.vertexes.keys())
            self.vertexes[keys[v_pos]].append(Edge(keys[w_pos], label, weight))
            if not self.__directed:
                self.vertexes[keys[w_pos]].append(Edge(keys[v_pos], label, weight))
        else:
            raise TypeError(f"V and W must have same types but are {type(v)} and {type(w)}.")

        self.__edges_counter += 1

    def search_vertex(self, v: (Vertex | str | int)) -> int:
        res = -1
        if type(v) == Vertex:    
            for i, j in enumerate(self.vertexes):
                if j == v:
                    res = i
                    break
        else:
            for i, j in enumerate(self.vertexes):
                if j.get_label() == v:
                    res = i
                    break
        return res
    
    @staticmethod
    def __search_edge_by_vertex_label(label: (str | int), l: list[Edge]) -> int:
        res = -1
        for i, j in enumerate(l):
            if j.to.get_label() == label:
                res = i
                break
        return res


    def search_edge(self, e: (Edge | str | int)) -> tuple:
        i = j = -1
        if type(e) == Edge:    
            for c1, v in enumerate(self.vertexes):
                for c2, ed in enumerate(self.vertexes[v]):
                    if e == ed:
                        i = c1
                        j = c2
                        break
        else:
            for c1, v in enumerate(self.vertexes):
                for c2, ed in enumerate(self.vertexes[v]):
                    if e == ed.get_label():
                        i = c1
                        j = c2
                        break
        return (i, j)

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
    
    def remove_vertex(self, v: (Vertex | str | int)) -> None:
        v_pos = self.search_vertex(v)
        if v_pos == -1:
            raise ValueError("The vertex V was not found in the graph.")
        
        if type(v) == Vertex:    
            self.vertexes.pop(v)
            
            # for each key, search for v in its values
            for iv in self.vertexes:
                for i, ed in enumerate(self.vertexes[iv]):
                    if v == ed.to:
                        self.vertexes[iv].pop(i)
        else:
            keys = list(self.vertexes.keys())
            self.vertexes.pop(keys[v_pos])

            # for each key, search for v in its values
            for iv in self.vertexes:
                for i, ed in enumerate(self.vertexes[iv]):
                    if v == ed.to.get_label():
                        self.vertexes[iv].pop(i)
        self.__n -= 1

    def remove_edge(self, e: (Edge | str | int)) -> None:
        pos = self.search_edge(e)

        w_pos = tmp = keys = None
        if type(e) == Edge:
            w_pos = self.search_vertex(e.to)
        else:
            # get edge from edge label and get 'to' attribute
            keys = list(self.vertexes.keys())
            tmp = self.search_edge(e)
            w_pos = self.search_vertex(self.vertexes[keys[tmp[0]]][tmp[1]].to)
        
        if pos[0] == -1:
            raise ValueError("The edge e was not found in the graph.")
        if w_pos == -1:
            raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
        
        v = list(self.vertexes.keys())[pos[0]]
        self.vertexes[v].pop(pos[1])

        if not self.__directed:
            if type(e) == Edge:
                e_pos = AdjacencyList.__search_edge_by_vertex_label(label=v.get_label(), l=self.vertexes[e.to])
                if e_pos == -1:
                    raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
                self.vertexes[e.to].pop(e_pos)
            else:
                e_pos = AdjacencyList.__search_edge_by_vertex_label(label=v.get_label(), l=self.vertexes[keys[w_pos]])
                if e_pos == -1:
                    raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
                self.vertexes[keys[w_pos]].pop(e_pos)
        self.__edges_counter -= 1

    def is_adjacent(self, v: (Vertex | str | int), w: (Vertex | str | int)) -> bool:
        # it does not matter if it is directed or not
        # if v is in w or w is in v, it is adjacent
        res = None

        v_pos = self.search_vertex(v)
        w_pos = self.search_vertex(w)
        if v_pos == -1:
            raise ValueError("Vertex V was not found in the Adjacency List.")
        if w_pos == -1:
            raise ValueError("Vertex W was not found in the Adjacency List.")
        
        if type(v) == type(w) == Vertex:
            res = AdjacencyList.__search_edge_by_vertex_label(label=w.get_label(), l=self.vertexes[v]) != -1\
            or AdjacencyList.__search_edge_by_vertex_label(label=v.get_label(), l=self.vertexes[w]) != -1
        elif type(v) == type(w) == str or type(v) == type(w) == int:
            keys = list(self.vertexes.keys())
            res = AdjacencyList.__search_edge_by_vertex_label(label=w, l=self.vertexes[keys[v_pos]]) != -1\
            or AdjacencyList.__search_edge_by_vertex_label(label=v, l=self.vertexes[keys[w_pos]]) != -1
        else:
            raise TypeError(f"V and W must have same types but are {type(v)} and {type(w)}.")
        return res

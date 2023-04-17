from .vertex.vertex import Vertex
from .edge.edge import Edge

class AdjacencyList:
    
    def __init__(self, n: int = 5, labels: tuple[(str | int)] = None,
                 weights: tuple[float] = None, directed: bool = False,
                 e_labeled: bool = False, e_weighted: bool = False) -> None:
        if n < 0:
            raise ValueError(f"N must be greater than or equal to zero, but received {n}.")
        
        self.__n = n
        self.__edges_counter = 0
        
        self.__directed = directed
        self.__vertexes = {} # {VertexOne: [EdgesOne], VertexTwo: [EdgesTwo]}

        self.__v_labeled = False if labels is None else True
        self.__v_weighted = False if weights is None else True

        if not self.__v_labeled:
            self.__vertex_iterable = 1

        self.__e_labeled = e_labeled
        self.__e_weighted = e_weighted

        if not self.__e_labeled:
            self.__edge_iterable = 1

        tmp_vertexes = self.__create_vertex(n, labels, weights)

        for v in tmp_vertexes:
            self.__vertexes[v] = []

    def size(self) -> int:
        """Returns the amount of vertexes in the graph."""
        return self.__n
    
    def edges(self) -> int:
        """Returns the amount of edges in the graph."""
        return self.__edges_counter
    
    def complete(self) -> bool:
        """If graph is simple, it is needed (n*(n-1))/2 edges for it to be complete.
        So, this method supposes this graph is simple."""
        return (self.__edges_counter == (self.__n * (self.__n - 1)) / 2)

    def v_labeled(self) -> bool:
        """Returns if the graph has labels for vertexes."""
        return self.__v_labeled
    
    def v_weighted(self) -> bool:
        """Returns if the graph has weights for vertexes."""
        return self.__v_weighted
    
    def e_labeled(self) -> bool:
        """Returns if the graph has labels for edges."""
        return self.__e_labeled
    
    def e_weighted(self) -> bool:
        """Returns if the graph has weights for edges."""
        return self.__e_weighted
    
    def empty(self) -> bool:
        """Returns if the amount of vertexes in the graph equals 0."""
        return self.__n == 0
    
    def print(self, verbose: bool = False) -> None:
        """Print the Adjacency List in the format 'v1 -> w1, w2, w3'"""

        print("Adjacency list vertexes\n")
        
        for v in self.__vertexes:
            print(v.to_string(verbose=verbose), " -> ", sep="", end="")
        
            if len(self.__vertexes[v]) > 0:
                print(self.__vertexes[v][0].to_string(verbose=verbose), sep="", end="")
        
                for e in self.__vertexes[v][1:]:
                    print(", ", e.to_string(verbose=verbose), sep="", end="")
            else:
                print("None", end="")
            print()
    
    def add_vertex(self, n: int,
                   label: (tuple[str] | tuple[int] | None) = None,
                   weight: (tuple[float] | None) = None) -> None:
        """It is supposed that if vertex is a tuple o vertexes, label and weight area also
        tuples with all the same length. If vertex is not a tuple, it is just a Vertex object,
        label is just a string or int or None and weight is just a float or None"""
        if n < 0:
            raise ValueError(f"N must be greater than or equal to zero, but received {n}.")
        
        if self.__v_labeled and self.__v_weighted:
            if label is not None and weight is not None and (not (n == len(label) == len(weight))):
                raise IndexError(f"The length for labels ({len(label)}) and weights ({len(weight)}) \
                                 cannot be None and must be equal to {n}.")
        elif self.__v_labeled:
            if label is not None and len(label) != n:
                raise IndexError(f"The length for labels ({len(label)}) cannot be None and must be equal to {n}.")
        elif self.__v_weighted:
            if weight is not None and len(weight) != n:
                raise IndexError(f"The length for weights ({len(weight)}) cannot be None and must be equal to {n}.")

        v = self.__create_vertex(n, label, weight)
        for i in v:
            self.__vertexes[i] = []
            self.__n += 1      

    def add_edge(self, v: (tuple[str] | tuple[int]), w: (tuple[str] | tuple[int]),
                 label: ((tuple[str] | tuple[int]) | None) = None, weight: (tuple[float] | None) = None) -> None:
        if len(v) != len(w):
            raise IndexError(f"The length of V ({len(v)}) and the length of W ({len(w)}) must be equal.")
        n = len(v)

        v_pos = []
        w_pos = []
        for i in range(len(v)):
            tmp1 = self.__search_vertex(v[i])
            tmp2 = self.__search_vertex(w[i])
            
            if tmp1 == -1:
                raise ValueError(f"The vertex {v[i]} was not found in the graph.")
            if tmp2 == -1:
                raise ValueError(f"The vertex {w[i]} was not found in the graph.")
            
            v_pos.append(tmp1)
            w_pos.append(tmp2)
        
        if self.__e_labeled and self.__e_weighted:
            if label is not None and weight is not None and (not (n == len(label) == len(weight))):
                raise IndexError(f"The length for labels ({len(label)}) and weights ({len(weight)}) \
                                 cannot be None and must be equal to {n}.")
        elif self.__e_labeled:
            if label is not None and len(label) != n:
                raise IndexError(f"The length for labels ({len(label)}) cannot be None and must be equal to {n}.")
        elif self.__e_weighted:
            if weight is not None and len(weight) != n:
                raise IndexError(f"The length for weights ({len(weight)}) cannot be None and must be equal to {n}.")

        keys = list(self.__vertexes.keys())
        all_ws = self.__create_edge(n, w, label, weight)

        for i in range(n):
            self.__vertexes[keys[v_pos[i]]].append(all_ws[i])
            if not self.__directed:
                self.__vertexes[keys[w_pos[i]]].append(Edge(keys[v_pos[i]], all_ws[i].get_label(), all_ws[i].get_weight()))

        self.__edges_counter += n
    
    def remove_vertex(self, v: (Vertex | str | int)) -> None:
        v_pos = self.__search_vertex(v)
        if v_pos == -1:
            raise ValueError(f"The vertex {v.get_label() if type(v) == Vertex else v} was not found in the graph.")
        
        v_to_w = w_to_v = 0

        if type(v) == Vertex:    
            v_to_w = len(self.__vertexes.pop(v))
            
            # for each key, search for v in its values
            for iv in self.__vertexes:
                for i, ed in enumerate(self.__vertexes[iv]):
                    if v == ed.to:
                        self.__vertexes[iv].pop(i)
                        w_to_v += 1
        else:
            keys = list(self.__vertexes.keys())
            v_to_w = len(self.__vertexes.pop(keys[v_pos]))
            
            # for each key, search for v in its values
            for iv in self.__vertexes:
                for i, ed in enumerate(self.__vertexes[iv]):
                    if v == ed.to.get_label():
                        self.__vertexes[iv].pop(i)
                        w_to_v += 1
        
        # subtract amount of edges
        self.__edges_counter -= (v_to_w + w_to_v) if self.__directed else v_to_w

        # subtract amount of vertexes
        self.__n -= 1

    def remove_edge(self, e: (Edge | str | int)) -> None:
        pos = self.__search_edge(e)

        if pos[0] == -1:
            raise ValueError(f"The edge {e.get_label() if type(e) == Edge else e} was not found in the graph.")
        
        w_pos = keys = None
        if type(e) == Edge:
            w_pos = self.__search_vertex(e.to)
        else:
            # get edge from edge label and get 'to' attribute
            keys = list(self.__vertexes.keys())
            w_pos = self.__search_vertex(self.__vertexes[keys[pos[0]]][pos[1]].to)
        
        if w_pos == -1:
            raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
        
        v = list(self.__vertexes.keys())[pos[0]]
        self.__vertexes[v].pop(pos[1])

        if not self.__directed:
            if type(e) == Edge:
                e_pos = AdjacencyList.__search_edge_by_vertex_label(label=v.get_label(), l=self.__vertexes[e.to])
                if e_pos == -1:
                    raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
                self.__vertexes[e.to].pop(e_pos)
            else:
                e_pos = AdjacencyList.__search_edge_by_vertex_label(label=v.get_label(), l=self.__vertexes[keys[w_pos]])
                if e_pos == -1:
                    raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
                self.__vertexes[keys[w_pos]].pop(e_pos)
        self.__edges_counter -= 1

    def is_adjacent(self, v: (Vertex | str | int), w: (Vertex | str | int)) -> bool:
        # it does not matter if it is directed or not
        # if v is in w or w is in v, it is adjacent
        res = None

        v_pos = self.__search_vertex(v)
        w_pos = self.__search_vertex(w)
        if v_pos == -1:
            raise ValueError(f"Vertex {v.get_label() if type(v) == Vertex else v} was not found in the Adjacency List.")
        if w_pos == -1:
            raise ValueError(f"Vertex {w.get_label() if type(w) == Vertex else w} was not found in the Adjacency List.")
        
        if type(v) == type(w) == Vertex:
            res = AdjacencyList.__search_edge_by_vertex_label(label=w.get_label(), l=self.__vertexes[v]) != -1\
            or AdjacencyList.__search_edge_by_vertex_label(label=v.get_label(), l=self.__vertexes[w]) != -1
        elif type(v) == type(w) == str or type(v) == type(w) == int:
            keys = list(self.__vertexes.keys())
            res = AdjacencyList.__search_edge_by_vertex_label(label=w, l=self.__vertexes[keys[v_pos]]) != -1\
            or AdjacencyList.__search_edge_by_vertex_label(label=v, l=self.__vertexes[keys[w_pos]]) != -1
        else:
            raise TypeError(f"V and W must have same types but are {type(v)} and {type(w)}.")
        return res
    
    def get_vertex(self, v_label: (str | int)) -> dict:
        v_pos = self.__search_vertex(v_label)
        if v_pos == -1:
            raise ValueError(f"Vertex {v_label} was not found in the Adjacency List.")
        v = list(self.__vertexes.keys())[v_pos]
        return {v: self.__vertexes[v]}
    
    def get_edge(self, e_label: (str | int)) -> Edge:
        e_pos = self.__search_edge(e_label)
        if e_pos[0] == -1:
            raise ValueError(f"Edge {e_label} was not found in the Adjacency List.")
        v = list(self.__vertexes.keys())[e_pos[0]]
        return self.__vertexes[v][e_pos[1]]

    def __create_vertex(self, n: int, labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None) -> dict:
        n_labels = tuple([i for i in range(self.__vertex_iterable, self.__vertex_iterable + n)]) if labels is None else labels
        n_weights = ((None,) * n) if weights is None else weights

        if labels is None:
            self.__vertex_iterable += n
        
        res = {}
        for i in range(n):
            res[Vertex(n_labels[i], n_weights[i])] = []

        return res
    
    def __create_edge(self, n: int, w: (tuple[str] | tuple[int]), labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None):
        if labels is not None and len(labels) != n:
            raise IndexError(f"The length for labels ({len(labels)}) must be equal to {n}.")
        if weights is not None and len(weights) != n:
            raise IndexError(f"The length for weights ({len(weights)}) must be equal to {n}.")
        
        n_labels = tuple([i for i in range(self.__edge_iterable, self.__edge_iterable + n)]) if labels is None else labels
        n_weights = ((None,) * n) if weights is None else weights

        if labels is None:
            self.__edge_iterable += n
        
        keys = list(self.__vertexes.keys())
        w_keys = []

        # get all w's
        for i in w:
            i_pos = self.__search_vertex(i)
            if i_pos == -1:
                raise ValueError(f"Vertex {i} was not found in the Adjacency List.")
            w_keys.append(keys[i_pos])

        return [Edge(w_keys[i], n_labels[i], n_weights[i]) for i in range(n)]

    def __search_vertex(self, v: (Vertex | str | int)) -> int:
        res = -1
        if type(v) == Vertex:    
            for i, j in enumerate(self.__vertexes):
                if j == v:
                    res = i
                    break
        else:
            for i, j in enumerate(self.__vertexes):
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


    def __search_edge(self, e: (Edge | str | int)) -> tuple:
        i = j = -1
        if type(e) == Edge:    
            for c1, v in enumerate(self.__vertexes):
                for c2, ed in enumerate(self.__vertexes[v]):
                    if e == ed:
                        i = c1
                        j = c2
                        break
        else:
            for c1, v in enumerate(self.__vertexes):
                for c2, ed in enumerate(self.__vertexes[v]):
                    if e == ed.get_label():
                        i = c1
                        j = c2
                        break
        return (i, j)

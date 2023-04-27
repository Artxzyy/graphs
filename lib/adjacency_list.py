from .vertex.vertex import Vertex
from .edge.edge import Edge

class AdjacencyList:
    """
    Adjacency List for the computational representation of a mathematical graph.
    It can have labels and weights in either edges or vertexes; it can be directed or not;
    it can be simple or not.

    Attributes

    n:
        The amount of vertexes in the graph. Must be greater than or equal to 0.
    edges_counter:
        The amount of edges in the graph. Must be greater than or equal to 0.
    directed:
        Defines if the edge methods should consider the operations in both ways or not.
    vertexes:
        Dictionary with all the vertexes and edges. All instances of the vertexes are keys in the dictionary,
        and each key have a list with the edges (and the edges have the edges info and the vertex it reaches).
    v_labeled:
        Defines if the vertexes should have user-defined labels.
    v_weighted:
        Defines if the vertexes should have user-defined weights.
    e_labeled:
        Defines if the edges should have user-defined labels.
    e_weighted:
        Defines if the edges should have user-defined weights.
    vertex_iterable and edge_iterable:
        If v_labeled is False, it is created a new attribute vertex_iterable as an internal labeling system.
        This being the reason all labels can be either strings or integers, if passed by the user, it is a string,
        if not, it is an integer. And the same goes for e_labeled, but with the edge_iterable attribute.
    """
    
    def __init__(self, n: int = 5, labels: tuple[(str | int)] = None,
                 weights: tuple[float] = None, directed: bool = False,
                 e_labeled: bool = False, e_weighted: bool = False) -> None:
        """
        AdjacencyList constructor

        Parameters

        n: int = 5
            starting number of vertexes
        labels: tuple[(str | int)] = None
            iterable with all labels for each vertex if the vertexes should be labeled
        weights: tuple[(str | int)] = None
            iterable with all weights for each vertex if the vertexes should be weighted
        directed: bool = False
            boolean that defines if the graph is directed or not
        e_labeled: bool = False
            boolean that defines if the edges should have labels
        e_weighted: bool = False
            boolean that defines if the edges should have weights

        """
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

    def info(self) -> dict:
        """Returns a dictionary with how this graph was built and how which parameters were defined."""
        return {
            "directed": self.__directed,
            "v_labeled": self.__v_labeled,
            "v_weighted": self.__v_weighted,
            "e_labeled": self.__e_labeled,
            "e_weighted": self.__e_weighted
        }

    def size(self) -> int:
        """Returns the amount of vertexes in the graph."""
        return self.__n
    
    def edges(self) -> int:
        """Returns the amount of edges in the graph."""
        return self.__edges_counter
    
    def complete(self) -> bool:
        """Tests if amount of edges is equal to ((n * (n - 1)) / 2). If equal, and supposing the graph is simple,
        returns True; otherwise, False."""
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
        """
        Print the Adjacency List in the format 
        
        'v1 -> w1, w2, w3.
         v2 -> w4, w5, w6.'
        
        Parameters

        verbose: bool = False
            boolean that defines if the vertexes will be printed with all information or just the label
        """

        for v in self.__vertexes:
            print(v.to_string(verbose=verbose), " -> ", sep="", end="")
        
            if len(self.__vertexes[v]) > 0:
                print(self.__vertexes[v][0].to.to_string(verbose=verbose), sep="", end="")
        
                for e in self.__vertexes[v][1:]:
                    print(", ", e.to.to_string(verbose=verbose), sep="", end="")
            else:
                print("None", end="")
            print()
    
    def add_vertex(self, n: int,
                   label: (tuple[str] | tuple[int] | None) = None,
                   weight: (tuple[float] | None) = None) -> None:
        """Add N vertexes with possible labels and weights passed as tuples.
        
        Parameters

        n: int
            Amount of vertexes to be added. Must be positive.
        label: (tuple[str] | tuple[int] | None) = None
            If v_labeled is True, label cannot be None and must be an iterable with size N.
            If v_labeled is False, label must be None.
        weight: (tuple[float] | None) = None
            If v_weighted is True, weight cannot be None and must be an iterable with size N.
            If v_weighted is False, weight must be None.
        """
        if n <= 0:
            raise ValueError(f"N must be greater than zero, but received {n}.")
        
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
        """Add N edges, N being len(v) and len(w) with possible labels and weights passed as tuples. The
        length for v, w, label (if e_labeled is True) and weight (if e_weighted is True) must all be equal. If
        the attribute directed is True, v is the vertex that the edge exit; if False, v and w can be switched and the
        result will be the same.
        
        Parameters

        v: (tuple[str] | tuple[int])
            The vertexes iterable that each edge will exit if the attribute directed is True; if
            false, v and w can be switched. Must have the same length as all other valid parameters.
        w: (tuple[str] | tuple[int])
            The vertexes iterable that each edge will reach if the attribute directed is True; if
            false, v and w can be switched. Must have the same length as all other valid parameters.
        label: (tuple[str] | tuple[int] | None) = None
            If e_labeled is True, label cannot be None and must have the same length as all other valid parameters.
            If e_labeled is False, label must be None.
        weight: (tuple[float] | None) = None
            If e_weighted is True, weight cannot be None and must have the same length as all other valid parameters.
            If e_weighted is False, weight must be None.
        """
        n = len(v)
        if n != len(w):
            raise IndexError(f"The length of V ({n}) and the length of W ({len(w)}) must be equal.")

        v_pos = []
        w_pos = []
        for i in range(len(v)):
            tmp1 = self.__search_vertex_position(v[i])
            tmp2 = self.__search_vertex_position(w[i])
            
            if tmp1 == -1 or tmp2 == -1:
                raise ValueError(f"The vertex {v[i]} was not found in the graph.")
            
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
        else:
            if label is not None:
                raise ValueError("The attribute e_labeled is False, but the label parameter is not None. With e_labeled False, label must be None.")
            if weight is not None:
                raise ValueError("The attribute e_weighted is False, but the weight parameter is not None. With e_weighted False, weight must be None.")
            
        keys = list(self.__vertexes.keys())
        all_ws = self.__create_edge(n, w, label, weight)

        for i in range(n):
            self.__vertexes[keys[v_pos[i]]].append(all_ws[i])
            if not self.__directed:
                self.__vertexes[keys[w_pos[i]]].append(Edge(keys[v_pos[i]], all_ws[i].get_label(), all_ws[i].get_weight()))

        self.__edges_counter += n
    
    def remove_vertex(self, v: (Vertex | str | int)) -> Vertex:
        """Remove vertex passed as parameter if possible.
        
            Parameters

            v: (Vertex | str | int)
                vertex to be removed. Can be the instance of the vertex or its label.
        """
        removed = None
        v_pos = self.__search_vertex_position(v)
        if v_pos == -1:
            raise ValueError(f"The vertex {v.get_label() if type(v) == Vertex else v} was not found in the graph.")
        
        v_to_w = w_to_v = 0

        if type(v) == Vertex:
            removed = self.__vertexes.pop(v)
            v_to_w = len(removed)
            
            # for each key, search for v in its values
            for iv in self.__vertexes:
                for i, ed in enumerate(self.__vertexes[iv]):
                    if v == ed.to:
                        self.__vertexes[iv].pop(i)
                        w_to_v += 1
        else:
            keys = list(self.__vertexes.keys())
            removed = self.__vertexes.pop(keys[v_pos])
            v_to_w = len(removed)
            
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

    def remove_edge(self, e: (Edge | str | int) = None, v: (str | int) = None, w: (str | int) = None) -> Edge:
        """Remove edge passed as parameter if possible. If e is passed, v and w are ignored.
        
            Parameters

            e: (Edge | str | int)
                Edge to be removed. Can be the instance of the edge or its label.
            v: (Vertex | str | int)
                Vertex the edge exit. Can be the instance of the vertex or its label.
            w: (Vertex | str | int)
                Vertex the edge reaches. Can be the instance of the vertex or its label.
        """
        removed = None

        pos = self.__search_edge_position(e, v, w)

        if pos[0] == -1:
            raise ValueError(f"The edge E was not found in the graph.")
        
        w_pos = keys = None
        if type(e) == Edge:
            w_pos = self.__search_vertex_position(e.to)
        else:
            # get edge from edge label and get 'to' attribute
            keys = list(self.__vertexes.keys())
            w_pos = self.__search_vertex_position(self.__vertexes[keys[pos[0]]][pos[1]].to)
        
        if w_pos == -1:
            raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
        
        ve = list(self.__vertexes.keys())[pos[0]]

        # removed edge
        removed = self.__vertexes[ve].pop(pos[1])

        if not self.__directed:
            if type(e) == Edge:
                e_pos = AdjacencyList.__search_edge_by_w_vertex_label(w_label=ve.get_label(), l=self.__vertexes[e.to])
                if e_pos == -1:
                    raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
                self.__vertexes[e.to].pop(e_pos)
            else:
                e_pos = AdjacencyList.__search_edge_by_w_vertex_label(w_label=ve.get_label(), l=self.__vertexes[keys[w_pos]])
                if e_pos == -1:
                    raise ValueError("The adjacency is not correctly formed. Something went wrong and we don't know what it is.")
                self.__vertexes[keys[w_pos]].pop(e_pos)
        self.__edges_counter -= 1

        return removed

    def is_adjacent(self, v: (Vertex | str | int), w: (Vertex | str | int)) -> bool:
        """Test if vertexes V and W are adjacent, meaning if there is an edge 'v -> w' or 'w -> v'.
        
        Parameters

        v: (Vertex | str | int)
            Vertex v. Can be the instance of the vertex or its label.
        w: (Vertex | str | int)
            Vertex w. Can be the instance of the vertex or its label.
        """
        res = None

        v_pos = self.__search_vertex_position(v)
        w_pos = self.__search_vertex_position(w)
        if v_pos == -1:
            raise ValueError(f"Vertex {v.get_label() if type(v) == Vertex else v} was not found in the Adjacency List.")
        if w_pos == -1:
            raise ValueError(f"Vertex {w.get_label() if type(w) == Vertex else w} was not found in the Adjacency List.")
        
        if type(v) == type(w) == Vertex:
            res = AdjacencyList.__search_edge_by_w_vertex_label(w_label=w.get_label(), l=self.__vertexes[v]) != -1\
            or AdjacencyList.__search_edge_by_w_vertex_label(w_label=v.get_label(), l=self.__vertexes[w]) != -1
        elif type(v) == type(w) == str or type(v) == type(w) == int:
            keys = list(self.__vertexes.keys())
            res = AdjacencyList.__search_edge_by_w_vertex_label(w_label=w, l=self.__vertexes[keys[v_pos]]) != -1\
            or AdjacencyList.__search_edge_by_w_vertex_label(w_label=v, l=self.__vertexes[keys[w_pos]]) != -1
        else:
            raise TypeError(f"V and W must have same types but are {type(v)} and {type(w)}.")
        return res
    
    def get_vertex(self, v_label: (str | int)) -> tuple:
        """Get the vertex and returns a tuple with the instance itself and the Adjacency List with its connections (edges)
        in the first and second positions of the tuple, respectively, if possible.
        
        Parameters

        v_label: (str | int)
            The label for the vertex to search.
        """
        v_pos = self.__search_vertex_position(v_label)
        if v_pos == -1:
            raise ValueError(f"Vertex {v_label} was not found in the Adjacency List.")
        v = list(self.__vertexes.keys())[v_pos]
        return (v, self.__vertexes[v])
    
    def get_edge(self, e_label: (str | int) = None,
                 v: (str | int) = None, w: (str | int) = None) -> Edge:
        """Get the edge and returns its instance if possible. If e_label is passed, v and w are ignored.
        If not, v and w cannot be None.
        
        Parameters

        e_label: (str | int)
            Label of edge to search. If None is passed, v and w are required.
        v: (str | int)
            Vertex the edge exit. If the attribute directed is False, v and w can be switched. It is ignored if e_label is not None.
        w: (str | int)
            Vertex the edge reaches. If the attribute directed is False, v and w can be switched. It is ignored if e_label is not None.
        """
        e_pos = self.__search_edge_position(e_label, v, w)
        if e_pos[0] == -1 or e_pos[1] == -1:
            raise ValueError(f"Edge {e_label} was not found in the Adjacency List.")
        ve = list(self.__vertexes.keys())[e_pos[0]]
        return self.__vertexes[ve][e_pos[1]]

    def update_vertex(self, v: (Vertex | str | int),
                      new_label: (str | int | None) = None, new_weight: (float | None) = None) -> None:
        # TODO update vertex
        pass
    
    def update_edge(self) -> None:
        # TODO update edge
        pass

    def __create_vertex(self, n: int, labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None) -> dict:
        """Create N vertexes with its labels and weights, if needed.
        
        Parameters

        n: int
            Amount of vertexes to create. It is supposed to be a positive number.
        labels: (tuple[str] | tuple[int] | None)
            Labels iterable. Must have length N if the attribute v_labeled is True; if False, the
            value must be None.
        weights: (tuple[float] | None)
            Weights iterable. Must have length N if the attribute v_weighted is True; if False, the
            value must be None.
        """
        if not self.__v_labeled and labels is not None:
            raise ValueError("The attribute v_labeled is False, but the labels parameter is not None. With v_labeled False, labels must be None.") 
        if not self.__v_weighted and weights is not None:
            raise ValueError("The attribute v_weighted is False, but the weights parameter is not None. With v_weighted False, weights must be None.")
        if labels is not None and len(labels) != n:
            raise IndexError(f"The length for labels ({len(labels)}) must be equal to {n}.")
        if weights is not None and len(weights) != n:
            raise IndexError(f"The length for weights ({len(weights)}) must be equal to {n}.")
        
        n_labels = tuple([i for i in range(self.__vertex_iterable, self.__vertex_iterable + n)]) if labels is None else labels
        n_weights = ((None,) * n) if weights is None else weights

        if labels is None:
            self.__vertex_iterable += n
        
        res = {}
        for i in range(n):
            res[Vertex(n_labels[i], n_weights[i])] = []

        return res
    
    def __create_edge(self, n: int, w: (tuple[str] | tuple[int]), labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None) -> list:
        """Create N edges with the attribute 'to' being each of W values with its labels and weights if needed.
        
        Parameters

        n: int
            Amount of edges to create. It is supposed to be a positive number.
        w: (tuple[str] | tuple[int])
            The iterable containing the vertexes each edge will reach. Must have length N.
        labels: (tuple[str] | tuple[int] | None)
            Labels iterable. Must have length N if the attribute e_labeled is True; if False, the
            value must be None.
        weights: (tuple[float] | None)
            Weights iterable. Must have length N if the attribute e_weighted is True; if False, the
            value must be None.
        """
        if not self.__e_labeled and labels is not None:
            raise ValueError("The attribute e_labeled is False, but the labels parameter is not None. With e_labeled False, labels must be None.") 
        if not self.__e_weighted and weights is not None:
            raise ValueError("The attribute e_weighted is False, but the weights parameter is not None. With e_weighted False, weights must be None.")
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
            i_pos = self.__search_vertex_position(i)
            if i_pos == -1:
                raise ValueError(f"Vertex {i} was not found in the Adjacency List.")
            w_keys.append(keys[i_pos])

        return [Edge(w_keys[i], n_labels[i], n_weights[i]) for i in range(n)]

    def __search_vertex_position(self, v: (Vertex | str | int)) -> int:
        """Returns the vertex position in the dictionary if it exists; -1 if not.
        
        Parameters
        
        v: (Vertex | str | int)
            Vertex to search. It can be the instance of the vertex or its label.
        """
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
    
    def __search_edge_list_by_v_vertex_label(self, v_label: (str | int)) -> tuple:
        """Get all edges adjacent to v_label and returns a tuple with v_label position
        and its values (adjacent edges) in the first and second position, respectively.
        
        Parameters

        v_label: (str | int)
            Vertex label to search.
        """
        i = self.__search_vertex_position(v_label)
        if i == -1:
            raise ValueError(f"Vertex {v_label} was not found in the Adjacency List.")
        return (i, self.__vertexes[list(self.__vertexes.keys())[i]])

    @staticmethod
    def __search_edge_by_w_vertex_label(w_label: (str | int), l: list[Edge]) -> int:
        """Search for a specific edge in a given list of edges using w_label to compare. Returns the position
        in the list i possible; -1 if not.
        
        Parameters

        w_label: (str | int)
            Label for vertex the edge reaches.
        l: list[Edge]
            List of edges to search
        """
        res = -1
        for i, j in enumerate(l):
            if j.to.get_label() == w_label:
                res = i
                break
        return res


    def __search_edge_position(self, e: (Edge | str | int) = None,
                      v: (str | int) = None, w: (str | int) = None) -> tuple:
        """Search for edge position in the vertexes attribute. If E is passed, v and w are ignored.
        If E is None, V and W cannot be None. This method returns a tuple with v and w positions in
        the first and second position, respectively, if found; (-1, -1) if not.
        
        Parameters

        e: (Edge | str | int) = None
            The edge to search position. It can be an instance of the edge or its label.
        v: (str | int) = None
            Label of the vertex the edge exit. Ignored if e is passed; required if not.
        w: (str | int) = None
            Label of the vertex the edge reaches. Ignored if e is passed; required if not.
        """
        i = j = -1

        if e is not None:
            if type(e) == Edge:
                for c1, ve in enumerate(self.__vertexes):
                    for c2, ed in enumerate(self.__vertexes[ve]):
                        if e == ed:
                            i = c1
                            j = c2
                            break
            else:
                for c1, ve in enumerate(self.__vertexes):
                    for c2, ed in enumerate(self.__vertexes[ve]):
                        if e == ed.get_label():
                            i = c1
                            j = c2
                            break
        elif v is None or w is None:
            raise ValueError("If E is not passed, it is required to pass V and W.")
        else:
            # search edge by label
            pos_and_values = self.__search_edge_list_by_v_vertex_label(v)
            i = pos_and_values[0]
            j = AdjacencyList.__search_edge_by_w_vertex_label(w, pos_and_values[1])
        return (i, j)
    
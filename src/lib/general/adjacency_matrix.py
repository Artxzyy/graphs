from .vertex.vertex import Vertex
from .edge.edge import Edge

class AdjacencyMatrix:
    """
    Adjacency Matrix for the computational representation of a mathematical graph.
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
        List with all the vertexes in order of creation.
    matrix:
        List of lists containing all the edges. The matrix[i][j] is the equivalent of getting
        the edge in the position matrix[vertexes[i]][vertexes[j]].
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
        AdjacencyMatrix constructor

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

        self.__v_labeled = False if labels is None else True
        self.__v_weighted = False if weights is None else True

        if not self.__v_labeled:
            self.__vertex_iterable = 1

        self.__e_labeled = e_labeled
        self.__e_weighted = e_weighted

        if not self.__e_labeled:
            self.__edge_iterable = 1

        tmp_vertexes = self.__create_vertex(n, labels, weights)

        self.__vertexes = [v for v in tmp_vertexes]
        self.__matrix = [[None] * n] * n

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
        len_v = len(v)
        self.__n += len_v

        for i in self.__matrix:
            for _ in range(len_v):
                i.append(None)

        for i in range(len_v):
            self.__vertexes.append(v[i])
            self.__matrix.append([None] * self.__n)

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
        
        edges = self.__create_edge(n, w, label, weight)

        for i in range(n):
            self.__matrix[v_pos[i]][w_pos[i]] = edges[i]
            if not self.__directed:
                self.__matrix[w_pos[i]][v_pos[i]] = edges[i]

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

        removed = self.__vertexes.pop(v_pos)

        v_to_w = w_to_v = 0

        for i in self.__matrix[v_pos]:
            if i is not None:
                v_to_w += 1

        self.__matrix.pop(v_pos)

        for i in self.__matrix:
            if i[v_pos] is not None:
                w_to_v += 1
            i.pop(v_pos)

        # subtract amount of edges
        self.__edges_counter -= (v_to_w + w_to_v) if self.__directed else v_to_w

        # subtract amount of vertexes
        self.__n -= 1

        return removed

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

        if pos[0] == -1 or pos[1] == -1:
            raise ValueError(f"The edge E was not found in the graph.")
        
        self.__matrix[pos[0]][pos[1]] = None
        if not self.__directed:
            self.__matrix[pos[1]][pos[0]] = None

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
        v_pos = self.__search_vertex_position(v)
        w_pos = self.__search_vertex_position(w)
        
        if v_pos == -1:
            raise ValueError(f"Vertex {v.get_label() if type(v) == Vertex else v} was not found in the Graph.")
        if w_pos == -1:
            raise ValueError(f"Vertex {w.get_label() if type(w) == Vertex else w} was not found in the Graph.")

        return self.__matrix[v_pos][w_pos] is not None or self.__matrix[w_pos][v_pos] is not None
    
    def get_vertex(self, v_label: (str | int)) -> Vertex:
        """Get the vertex and returns it.
        
        Parameters

        v_label: (str | int)
            The label for the vertex to search.
        """
        v_pos = self.__search_vertex_position(v_label)

        if v_pos == -1:
            raise ValueError(f"Vertex {v_label} was not found in the Graph.")

        return self.__vertexes[v_pos]
    
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
            raise ValueError(f"Edge {e_label} was not found in the Graph.")
        
        return self.__matrix[e_pos[0]][e_pos[1]]
    
    def update_vertex(self, v: (Vertex | str | int),
                      new_label: (str | int | None) = None, new_weight: (float | None) = None) -> None:
        """Update vertex data with new_label and new_weight if the graph is v_labeled or v_weighted, respectively.
        If the graph is v_labeled, new_label cannot be None and if the graph is v_weighted, new_weight cannot be None.
        If new_label already exists in any vertex in the graph, including in vertex v, the method will raise a ValueError.

        Parameters

        v: (Vertex | str | int)
            Vertex to be updated.
        new_label: (str | int | None) = None
            New label for the vertex, if the graph is labeled. If new_label is either equal to v label or is None,
            a ValueError will be raised.
        new_weight: (float | None) = None
            New weight for the vertex, if the graph is weighted. If new_weight is None, a ValueError will be raised.
        """
        v_pos = self.__search_vertex_position(v)
        if v_pos == -1:
            raise ValueError(f"Edge {v.get_label() if type(v) == Vertex else v} was not found in the Graph.")
        
        vertex: Vertex = self.__vertexes[v_pos]
        if self.__v_labeled:
            if new_label is None:
                raise ValueError("New label cannot be None.")
            if self.__v_label_exists(new_label):
                raise ValueError(f"The label '{new_label}' already exists.")
            vertex.set_label(new_label)
        if self.__v_weighted:
            if new_weight is None:
                raise ValueError("New weight cannot be None.")
            vertex.set_weight(new_weight)

    def update_edge(self, e: (Edge | str | int | None) = None,
                    v: (Vertex | str | int | None) = None,  w: (Vertex | str | int | None) = None,
                    new_label: (str | int | None) = None, new_weight: (float | None) = None) -> None:
        """Update edge data with new_label and new_weight if the graph is e_labeled or e_weighted, respectively.
        If the graph is e_labeled, new_label cannot be None and if the graph is e_weighted, new_weight cannot be None.
        If new_label already exists in any edge in the graph, including in edge e, the method will raise a ValueError.

        Parameters

        e: (Edge | str | int | None) = None
            Vertex to be updated.
        v: (Vertex | str | int | None) = None
            Vertex the edge exit. If the attribute directed is False, v and w can be switched. It is ignored if e_label is not None.
        w: (Vertex | str | int | None) = None
            Vertex the edge reaches. If the attribute directed is False, v and w can be switched. It is ignored if e_label is not None.
        new_label: (str | int | None) = None
            New label for the vertex, if the graph is labeled. If new_label is either equal to e label or is None,
            a ValueError will be raised.
        new_weight: (float | None) = None
            New weight for the vertex, if the graph is weighted. If new_weight is None, a ValueError will be raised.
        """
        e_pos = self.__search_edge_position(e, v, w)

        if e_pos[0] == -1 or e_pos[1] == -1:
            raise ValueError(f"This edge was not found in the Graph.")
        
        e: Edge = self.__matrix[e_pos[0]][e_pos[1]]
        if self.__e_labeled:
            if new_label is None:
                raise ValueError("New label cannot be None.")
            if self.__e_label_exists(new_label):
                raise ValueError(f"The label '{new_label}' already exists.")
            e.set_label(new_label)
        if self.__e_weighted:
            if new_weight is None:
                raise ValueError("New weight cannot be None.")
            e.set_weight(new_weight)

    def __create_vertex(self, n: int, labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None) -> list:
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
        
        return tuple([Vertex(n_labels[i], n_weights[i]) for i  in range(n)])
    
    def __create_edge(self, n: int, w: (tuple[str] | tuple[int]), labels: (tuple[str] | tuple[int] | None) = None,
                      weights: (tuple[float] | None) = None) -> list:
        """Create N edges with its labels and weights if needed.
        
        Parameters

        n: int
            Amount of edges to create. It is supposed to be a positive number.
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
    
        return tuple([Edge(n_labels[i], n_weights[i]) for i in range(n)])
    
    def __search_vertex_position(self, v: (Vertex | str | int)) -> int:
        """Returns the vertex position in the vertexes list if it exists; -1 if not.
        
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
                for c1, row in enumerate(self.__matrix):
                    for c2, column in enumerate(row):
                        if e == column:
                            i = c1
                            j = c2
                            break
            else:
                for c1, row in enumerate(self.__matrix):
                    for c2, column in enumerate(row):
                        if e == column.get_label():
                            i = c1
                            j = c2
                            break
        elif v is None or w is None:
            raise ValueError("If E is not passed, it is required to pass V and W.")
        else:
            v_pos = self.__search_vertex_position(v)
            w_pos = self.__search_vertex_position(w)

            if v_pos == -1:
                raise ValueError(f"The vertex {v.get_label() if type(v) == Vertex else v} was not found in the graph.")
            if w_pos == -1:
                raise ValueError(f"The vertex {w.get_label() if type(w) == Vertex else w} was not found in the graph.")
        
            i = v_pos
            j = w_pos

        return (i, j)
    
    def __v_label_exists(self, label: (str | int)) -> bool:
        """Search if given label exists in any vertex in the graph
        
        Parameters
        
        label: (str | int)
            Label to search in the graph vertexes.
        """
        res = False
        for v in self.__vertexes:
            if v.get_label() == label:
                res = True
                break
        return res
    
    def __e_label_exists(self, label: (str | int)) -> bool:
        """Search if given label exists in any edge in the graph
        
        Parameters
        
        label: (str | int)
            Label to search in the graph edges.
        """
        res = False
        for i in self.__matrix:
            for j in i:
                if label == j.get_label():
                    res = True
                    break
        return res
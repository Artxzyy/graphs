from list_nodes import AllWeightedAdjacencyNode, VWeightedAdjacencyNode, EWeightedAdjacencyNode, AdjacencyNode

class Graph:
    """ Adjacency List Graph """

    def __init__(self, n: int, vertex_weighted: bool = False, edge_weighted: bool = False) -> None:
        self.n = n

        # instanciate adjacency list with n null values, if negative or 0 it creates an empty list
        self.graph = [None] * self.n

        self.node_class = None
        if vertex_weighted and edge_weighted:
            self.node_class = AllWeightedAdjacencyNode
        elif vertex_weighted:
            self.node_class = VWeightedAdjacencyNode
        elif edge_weighted:
            self.node_class = EWeightedAdjacencyNode
        else:
            self.node_class = AdjacencyNode




g = Graph(n=10, vertex_weighted=True)

print(g.graph)
g.graph[0] = g.node_class(1, vertex_weight=10)

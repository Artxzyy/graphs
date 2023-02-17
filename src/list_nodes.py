class AdjacencyNode:
    """ Default no-weighted adjacency list node """

    def __init__(self, vertex: (int | str), next=None) -> None:
        self.vertex = vertex
        self.next = next


class VWeightedAdjacencyNode(AdjacencyNode):
    """ Vertex weighted adjacency list node """

    def __init__(self, vertex: (int | str), next=None, vertex_weight: float = 1.0) -> None:
        super().__init__(vertex, next)

        self.v_weight = vertex_weight


class EWeightedAdjacencyNode(AdjacencyNode):
    """ Edge weighted adjacency list node """

    def __init__(self, vertex: (int | str), next=None, edge_weight: float = 1.0) -> None:
        super().__init__(vertex, next)

        self.e_weight = edge_weight


class AllWeightedAdjacencyNode(AdjacencyNode):
    """ Vertex and edge weighted adjacency list node """

    def __init__(self, vertex: (int | str), next=None, edge_weight: float = 1.0) -> None:
        super().__init__(vertex, next)

        self.v_weight = edge_weight
        self.e_weight = edge_weight
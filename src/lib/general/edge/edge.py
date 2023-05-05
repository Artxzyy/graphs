class Edge:
    def __init__(self, label: (str | int) = None, weight: float = None) -> None:
        self.__label = label
        self.__weight = weight

    def get_label(self) -> (str | int):
        return self.__label
    
    def set_label(self, label: (str | int)) -> None:
        self.__label = label

    def get_weight(self) -> float:
        return self.__weight
    
    def set_weight(self, weight) -> None:
        self.__weight = weight

    def to_string(self, verbose: bool = False) -> str:
        return f"(label: '{self.__label}', weight: '{self.__weight}')" if verbose else f"{self.__label}"
    
class ListEdge(Edge):

    def __init__(self, to, label: (str | int) = None, weight: float = None) -> None:
        super().__init__(label=label, weight=weight)
        self.to = to
    
    def to_string(self, verbose: bool = False) -> str:
        return f"(to vertex: '{self.to.to_string()}', label: '{self.__label}', weight: '{self.__weight}')" if verbose else f"{self.__label}"
    
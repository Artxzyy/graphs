class Vertex:
    def __init__(self, label: (str | int) = None, weight: float = None, iterator: int = 1) -> None:
        self.__label = iterator if label is None else label
        self.__weight = weight

    def get_label(self) -> (str | int):
        return self.__label
    
    def set_label(self, label: (str | int)) -> None:
        self.__label = label

    def get_weight(self) -> float:
        return self.__weight
    
    def set_weight(self, weight: float) -> None:
        self.__weight = weight
    
    def to_string(self, verbose: bool = False) -> str:
        return f"(label: '{self.__label}', weight: '{self.__weight}')" if verbose else f"{self.__label}"
    
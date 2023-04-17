class Vertex:
    def __init__(self, label: (str | int) = None, weight: float = None, iterator: int = 1) -> None:
        self.__label = iterator if label is None else label
        self.__weight = weight

    def get_label(self) -> (str | int):
        return self.__label
    
    def set_label(self, new_label: (str | int), others: tuple) -> None:
        if new_label in others:
            raise ValueError((f"This vertex '{self.__label}' cannot be updated. The id '{new_label}' already exists."))
        else:
            self.__label = new_label

    def get_weight(self) -> float:
        return self.__weight
    
    def to_string(self, verbose: bool = False) -> str:
        return f"(label: '{self.__label}', weight: '{self.__weight}')" if verbose else f"{self.__label}"
    
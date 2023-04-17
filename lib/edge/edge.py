class Edge:
    def __init__(self, to, label: (str | int) = None, weight: float = None) -> None:
        self.to = to
        self.__label = label
        self.__weight = weight

    def get_label(self) -> (str | int):
        return self.__label
    
    def set_label(self, new_label: (str | int), others: tuple) -> None:
        if new_label in others:
            raise ValueError((f"This vertex '{self.__label}' cannot be updated. The id '{new_label}' already exists."))
        else:
            self.__label = new_label

    def get_weight(self) -> (float):
        return self.__weight
    
    def to_string(self, verbose: bool = False) -> str:
        return f"(to vertex: '{self.to.to_string()}', label: '{self.__label}', weight: '{self.__weight}')" if verbose else f"{self.__label}"
    
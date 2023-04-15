class Edge:
    def __init__(self, to, label: (str | int) = None, weight: float = None) -> None:
        self.to = to
        self.__label = label
        self.weight = weight

    def get_label(self) -> (str | int):
        return self.__label
    
    def set_label(self, new_label: (str | int), others: tuple) -> None:
        if new_label in others:
            raise ValueError((f"This vertex '{self.__label}' cannot be updated. The id '{new_label}' already exists."))
        else:
            self.__label = new_label
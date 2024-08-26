from enum import Enum

class Shape(Enum):
    CRUZ = "X"
    CIRC = "O"
    NADA = "-"


class Tateti:
    def __init__(self) -> None:
        self.mat = [
            [Shape.NADA, Shape.NADA,Shape.NADA],
            [Shape.NADA, Shape.NADA,Shape.NADA],
            [Shape.NADA, Shape.NADA,Shape.NADA]
            ]
    def setVal(self, val: Shape, x: int, y: int):
        if self.isFree(x, y):
            self.mat[x][y]
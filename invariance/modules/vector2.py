from __future__ import annotations


class Vector2:
    def __init__(self, y: int, x: int) -> None:
        self.y: int = y
        self.x: int = x

    @property
    def xy(self) -> Vector2:
        return Vector2(self.x, self.y)

    @property
    def yx(self) -> Vector2:
        return Vector2(self.y, self.x)

    def __add__(self, other: Vector2) -> Vector2:
        return Vector2(self.y + other.y, self.x + other.x)

    def __sub__(self, other: Vector2) -> Vector2:
        return Vector2(self.y - other.y, self.x - other.x)

    def __neg__(self):
        return Vector2(-self.y, -self.x)

    def __str__(self) -> str:
        return f"({self.y}, {self.x})"

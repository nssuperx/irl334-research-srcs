from typing import Tuple
import itertools
import torch
from torch.utils.data import Dataset


class VerticalLine(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pattern = (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1))
        choices = itertools.product(range(4), repeat=3)
        correct_pattern_choice = ((1, 1, 1), (2, 2, 2), (3, 3, 3))
        self.data: list[tuple[torch.Tensor, int]] = []

        for c in choices:
            array_tmp = []
            for i in c:
                array_tmp.append(pattern[i])
            self.data.append((torch.Tensor(array_tmp), 1 if c in correct_pattern_choice else 0))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index][0], self.data[index][1]

    def __len__(self) -> int:
        return len(self.data)


class XORProblem(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = [
            (torch.Tensor([0.0, 0.0]), 0),
            (torch.Tensor([1.0, 0.0]), 1),
            (torch.Tensor([0.0, 1.0]), 1),
            (torch.Tensor([0.0, 0.0]), 0)
        ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index][0], self.data[index][1]

    def __len__(self) -> int:
        return len(self.data)

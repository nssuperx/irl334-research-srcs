import itertools
import torch

pattern = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1))
choices = itertools.product(range(4), repeat=3)

correct_pattern_choice = ((1, 1, 1), (2, 2, 2), (3, 3, 3))

data = []

for c in choices:
    array_tmp = []
    for i in c:
        array_tmp.append(pattern[i])
    data.append([torch.Tensor(array_tmp), torch.Tensor([1 if c in correct_pattern_choice else 0])])

for d in data:
    print(d)

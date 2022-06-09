import numpy as np

a = np.ones((2, 3))
print(a)

print(np.pad(a, 3))
print(np.pad(a, (2, 3)))
# axis=0, axis=1で指定
print(np.pad(a, ((1, 2), (3, 4))))

import numpy as np
from sklearn.decomposition import NMF

V = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12]])

print("--- matrix V ---")
print(V)

nmf = NMF(n_components=3)
nmf.fit(V)

print("--- matrix W ---")
W = nmf.fit_transform(V)
print(W)

print("--- matrix H ---")
H = nmf.components_
print(H)

print("--- matrix WH ---")
WH = np.dot(W, H)
print(WH)

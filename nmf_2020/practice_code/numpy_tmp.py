import numpy as np

V = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print("V.shape:" + str(V.shape))
W = np.array([[1,2],[3,4],[5,6],[7,8]])
print("W.shape:" + str(W.shape))
H = np.array([[1,2,3],[4,5,6]])
print("H.shape:" + str(H.shape))
WH = np.dot(W,H)
print("WH.shape:" + str(WH.shape))

import numpy as np

a = np.random.random((4, 4))
b = np.random.random((4, 4))

print(f"a: {a}\nb: {b}")

print(f"a id: {id(a)}")
print(f"b id: {id(b)}")

c = a

print(f"c id: {id(c)}")
print(f"a == c: {id(a) == id(c)}")
print(f"shares_memory : {np.shares_memory(a, c)}")


print(f"a[0, 0] id: {id(a[0, 0])}")

d = a[0:2, 0:2]
print(d)
print(f"d id: {id(d)}")
print(f"d type: {type(d)}")
print(f"d shape: {d.shape}")

e = a[0:2, 0:2]
print(e)
print(f"e id: {id(e)}")
print(f"e type: {type(e)}")
print(f"e shape: {e.shape}")

print(f"d == e: {id(d) == id(e)}")
print(f"shares_memory : {np.shares_memory(d, e)}")

print(f"a[0:2, 0:2] id: {id(a[0:2, 0:2])}")
print(f"a[0:2, 0:2] type: {type(a[0:2, 0:2])}")

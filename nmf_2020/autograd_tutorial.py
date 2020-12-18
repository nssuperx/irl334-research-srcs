import torch

# テンソルxを作成．全要素「1」
# 勾配を記録する．requires_grad=True
x = torch.ones(2, 2, requires_grad=True)
print(x)

# xの全要素に2を足したものをyとする．
# このときyもrequires_grad=Trueとなっている．
y = x + 2
print(y)

# zとoutを作成．
z = y * y * 3
out = z.mean()
print(z)
print(out)

# tensor.requires_grad_()で，勾配を記録するか設定できる．
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 返り値None
print(x.grad)

# 勾配を計算．返り値None
print(out.backward())

# outのxの偏微分を表示
# d(out)/dx
print(x.grad)


x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# 勾配保持の確認
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

# torch.detach()
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

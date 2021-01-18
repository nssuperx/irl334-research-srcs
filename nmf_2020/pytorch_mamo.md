# Pytorchのメモ

## Tensor型のメソッドの，`.sub_` の `_` はなに．
In-place version of sub()

[リファレンス](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.sub_)

`_` が，ついてないメソッドのIn-place version.

In-placeとは，その場で，という意味．
[In-placeアルゴリズム](https://ja.wikipedia.org/wiki/In-place%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0)に書いてあるとおりだと思う．

`_` がついてないメソッドは，計算後の値をreturnするだけ．`_` がついてるのは，元のデータを計算後の値に更新かつ，その値をreturnする．

## torch.no_grad()とは
with文と組み合わせて使う．
このブロックの中で行う計算は，勾配を考慮しない値の計算ができる．  
通常は，Tensor型で `requires_grad=True` なデータを計算するときは，逆伝播できるように勾配も考慮して計算される．（たぶん）

### **補足** with文とは
開始と終了がセットになった処理を行うときに使う．  
`open()` などで使う．これらには処理が（ブロックが？）始まるときと終了するときに呼び出すメソッドが定義されている．

* `def __enter__(self):`
* `def __exit__(self, ex_type, ex_value, trace):`

など．

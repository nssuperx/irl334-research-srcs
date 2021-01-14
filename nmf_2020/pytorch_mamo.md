# Pytorchのメモ

## Tensor型のメソッドの，`.sub_` の `_` はなに．
In-place version of sub()

[リファレンス](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.sub_)

`_` が，ついてないメソッドのIn-place version.

In-placeとは，その場で，という意味．
[In-placeアルゴリズム](https://ja.wikipedia.org/wiki/In-place%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0)に書いてあるとおりだと思う．

`_` がついてないメソッドは，計算後の値をreturnするだけ．`_` がついてるのは，元のデータを計算後の値に更新かつ，その値をreturnする．

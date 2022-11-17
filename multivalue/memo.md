# メモなど

長いコメントを残したいが，ソースに書きたくないのでこっちに書く．

## `HiddenBrick`の`nn.Linear`について

[play.py](./play.py)の`HiddenBrick`モジュールの注意

ここの全結合層について．
nn.Linear自体はN次元の入力を扱えるようになっている．
つまり入力の末端の次元がLinearのin_featuresとあっていればよい．
しかし，ソースとかリファレンス見た感じ，weightは2次元．
各次元ごとに重みを持っているわけではない．
なので今回の実験のような，同じ入力で重みの異なる線形結合をたくさんしたい，みたいな用途では使えない．
そのため，素子数 * 出力数の全結合を作ってあとからreshapeする．
nn.ConvNdをうまく使えそうだが，今の実力ではよくわからなかった．
もっといい実装は今後の課題とする．

## 微分可能なArgmax

[このコード](https://github.com/david-wb/softargmax/blob/master/softargmax.py)を使わせてもらう．

## `torchvision.utils.make_grid()`

引数について
> tensor (Tensor or list) – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.

おそらくBatch, Channel, Height, Weight．

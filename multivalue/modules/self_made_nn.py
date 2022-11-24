import torch
from torch import nn


class TSigmoid(nn.Module):
    """調整できるsigmoid関数
    Args:
        T (float, optional): 歪ませる度合い. Defaults to 1.0.
        center (float, optional): 真ん中の位置(出力が0.5になる点). Defaults to 0.0.
    """

    def __init__(self, T: float = 1.0, center: float = 0.0) -> None:
        super(TSigmoid, self).__init__()
        self.Ti = 1.0 / T  # T inverse
        self.center = center

    def forward(self, input: torch.Tensor):
        return torch.sigmoid((input - self.center) * self.Ti)


class SoftArgmax(nn.Module):
    """微分可能なArgmax
    NOTE: 参考元 https://github.com/david-wb/softargmax/blob/master/softargmax.py
    """

    def __init__(self, beta=100) -> None:
        super(SoftArgmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.beta = beta

    def forward(self, input: torch.Tensor):
        *_, n = input.shape
        input = self.softmax(self.beta * input)
        indices = torch.linspace(0, 1, n)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result


class ClampArg(nn.Module):
    """出力番号（要素）を0, 1の間でクランプする
    0は0，1以上は1にできる
    NOTE: 必要ないかもしれない
    """

    def __init__(self):
        super(ClampArg, self).__init__()

    def forward(self, input: torch.Tensor):
        return input.clamp(min=0, max=1)


class MultiValueBrick(nn.Module):
    """隠れ層の役割のBrick
    """

    def __init__(self, in_features: int, out_features: int, classes: int):
        super(MultiValueBrick, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.classes = classes
        self.flatten = nn.Flatten()
        # NOTE: メモ参照．HiddenBrickのnn.Linearについて
        self.fc = nn.Linear(self.in_features, self.classes * self.out_features)
        self.argmax = SoftArgmax()
        # self.clamp = ClampArg()
        self.tsigmoid = TSigmoid(T=0.1, center=0.5)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.out_features, self.classes)
        x = self.argmax(x)
        # x = self.clamp(x)
        x = self.tsigmoid(x)
        return x


class OutBrick(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(OutBrick, self).__init__()
        # NOTE: 「該当なし」を考慮する(+1する)のは，ネットワークまたは使う側か考える
        self.fc = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.fc(x.to(dtype=torch.float32))
        x = self.softmax(x)
        return x

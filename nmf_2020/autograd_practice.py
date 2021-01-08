import torch
import matplotlib.pyplot as plt

"""
やりたいこと
z = x * y
z = 12
求めたいもの, x = 3 y = 4
xとyを当てる
"""

mu = 0.01
iteration = 1000

def main():
    target_z = torch.tensor(12, requires_grad=False)
    target_x = torch.tensor(3, requires_grad=False)
    target_y = torch.tensor(4, requires_grad=False)

    x = torch.rand(1, requires_grad=True)
    y = torch.rand(1, requires_grad=True)

    print("x:" + str(x))
    print("y:" + str(y))

    # klDivLoss = torch.nn.KLDivLoss()

    # ログ用リスト
    x_LOG = []
    y_LOG = []
    dist_LOG = []

    for i in range(iteration):
        # 距離を計算
        distance = torch.dist(target_z, x * y)
        # distance = klDivLoss(target_z, x * y)
        dist_LOG.append(distance.data)

        # 微分
        distance.backward()

        x_LOG.append(x.data.clone())
        y_LOG.append(y.data.clone())

        # 引く（勾配の向きにずらす）
        x.data.sub_(mu * x.grad.data)
        y.data.sub_(mu * y.grad.data)

        # 微分をゼロに．ここよくわからない．
        x.grad.data.zero_()
        y.grad.data.zero_()

        if((i+1) % 10 == 0):
            print("iter:" + str(i) + "   x:"+ str(x.data) + "   y:"+ str(y.data))

    plt.plot(x_LOG, y_LOG)
    plt.show()
    plt.plot(range(iteration), dist_LOG)
    plt.show()
        


if __name__ == "__main__":
    main()

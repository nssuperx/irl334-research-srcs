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
iteration = 100

klDivLoss = torch.nn.KLDivLoss()

def main():
    target_z = torch.tensor(12, requires_grad=False)
    target_x = torch.tensor(3, requires_grad=False)
    target_y = torch.tensor(4, requires_grad=False)

    for start_x in range(15):
        for start_y in range(15):

            x = torch.tensor(float(start_x), requires_grad=True)
            y = torch.tensor(float(start_y), requires_grad=True)

            # print("x:" + str(x))
            # print("y:" + str(y))

            # ログ用リスト
            x_LOG = []
            y_LOG = []

            for i in range(iteration):
                # 距離を計算
                # distance = torch.dist(target_z, x * y)
                distance = torch.sqrt((target_z - x * y)**2)
                # distance = klDivLoss(target_z, x * y)
                # distance = kl_divergence(target_z, x, y)

                # 微分
                distance.backward()

                x_LOG.append(x.data.clone())
                y_LOG.append(y.data.clone())

                # 引く（勾配の向きにずらす）
                # x.data.sub_(mu * x.grad.data)
                # y.data.sub_(mu * y.grad.data)
                x.data.mul_(torch.exp(-mu * x.grad.data))
                y.data.mul_(torch.exp(-mu * y.grad.data))

                # 微分をゼロに．ここよくわからない．
                x.grad.data.zero_()
                y.grad.data.zero_()

                """
                if((i+1) % 10 == 0):
                    print("iter:" + str(i) + "   x:"+ str(x.data) + "   y:"+ str(y.data))
                """

            plt.plot(x_LOG, y_LOG)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def kl_divergence(V, W, H):
    WH = W * H
    # F = V * torch.log(WH) - WH
    F = V * torch.log(V / WH) - V + WH
    # F = V * torch.log(V / WH)
    return F

if __name__ == "__main__":
    main()

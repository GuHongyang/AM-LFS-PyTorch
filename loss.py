import torch
import torch.nn as nn
import torch.nn.functional as F


def t_fun(x, theta, x_range = [-1, 1]):
    M = theta.size(0)// 2

    interval = (x_range[1] - x_range[0]) / M
    y = 0
    for m in range(M):
        if m == M - 1:
            ind = (x >= (x_range[0] + interval * m)).float()
        elif m == 0:
            ind = (x < (x_range[0] + interval * (m + 1))).float()
        else:
            ind = ((x >= (x_range[0] + interval * m)).float() * (x < (x_range[0] + interval * (m + 1))).float())
        y += (x * theta[m] + theta[m+M]) * ind

    return y


# def loss(x, y, M, theta):
#     cos_x = x[0]
#     weight_norm = x[1]
#     theta = theta.to(cos_x.device).detach()
#     y_onehot = F.one_hot(y, 10)
#     t_cos_x = t_fun(cos_x, theta, x_range=torch.Tensor([-1, 1]).to(cos_x.device))
#     loss = -F.log_softmax((t_cos_x * y_onehot + cos_x * (1 - y_onehot))*weight_norm, dim=1)[torch.arange(cos_x.size(0)).to(cos_x.device), y]
#     return loss.mean()

def loss(cos_x, y, M, theta):
    theta = theta.to(cos_x.device).detach()
    y_onehot = F.one_hot(y, 10)

    t1_cos_x = t_fun(cos_x, theta[:2 * M], x_range=torch.Tensor([-1, 1]).to(cos_x.device))
    # t2_cos_x = t_fun(cos_x, theta[2*M:], x_range=torch.Tensor([-1,1]).to(cos_x.device))
    # p_y_t = F.softmax(t1_cos_x * y_onehot + t2_cos_x * (1 - y_onehot), dim=1)[torch.arange(cos_x.size(0)).to(cos_x.device), y]

    # tao_p_y_t = t_fun(p_y_t, theta[2 * M:], x_range=torch.Tensor([0, 1]).to(cos_x.device))
    loss = -F.log_softmax((t1_cos_x * y_onehot + cos_x * (1 - y_onehot)), dim=1)[
        torch.arange(cos_x.size(0)).to(cos_x.device), y]
    return loss.mean()


if __name__ == '__main__':

    a = torch.rand([10,5])*2 -1

    theta = torch.Tensor([1,2,3,0,0,0])
    print(a)
    print(t_fun(a, theta, torch.Tensor([-1,1])))


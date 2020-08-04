import torch
from torch import nn


def mean_std(feature):

    size = feature.size()
    n, c = size[:2]

    mean = torch.mean(feature.view(n, c, -1), 2).view(n, c, 1, 1)
    std = torch.std(feature.view(n, c, -1), 2).view(n, c, 1, 1) + 1e-10

    return mean, std


def StyleLoss(s_feature, r_feature):

    loss = nn.MSELoss()

    r_mean, r_std = mean_std(r_feature)
    s_mean, s_std = mean_std(s_feature)

    s_loss = loss(s_mean, r_mean) + loss(s_std, r_std)

    return s_loss

import torch
import torch.nn as nn
import torch.nn.functional as ff


def l1_loss(input, target, **kwargs):
    return ff.l1_loss(input, target, **kwargs)

def mse_loss(input, target, **kwargs):
    return ff.mse_loss(input, target, **kwargs)

def l2_loss(input, target, **kwargs):
    return torch.square(ff.mse_loss(input, target, **kwargs))


if __name__ == '__main__':

    B, S, C, L = 10, 4, 2, 100000
    inputs = torch.randn(B, C, S, L)
    targets = torch.randn(B, C, S, L)

    loss = l1_loss(inputs, targets, reduction='sum')
    print(f'L1 loss: {loss.item():.3f}')

    loss = mse_loss(inputs, targets, reduction='sum')
    print(f'MSE loss: {loss.item():.3f}')

    loss = l2_loss(inputs, targets, reduction='sum')
    print(f'L2 loss: {loss.item():.3f}')

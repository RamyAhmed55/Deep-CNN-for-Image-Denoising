# loss.py

from torch import nn


def build_loss():
    return nn.MSELoss(reduction="sum")
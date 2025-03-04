import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    flipped = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return flipped, targets

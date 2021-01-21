from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg

logabs = lambda x: torch.log(torch.abs(x))

class Glow(nn.Module):
    def __init__(self, in_channels: int, n_flows: int, n_blocks: int, affine: bool = True, conv_lu: bool = True):
        super().__init__()

        # Create set of standard Glow blocks
        self.blocks = nn.ModuleList(
            [Block(2 * i * in_channels, n_flows, affine, conv_lu) for i in range(1, n_blocks)]
        )

class Block(nn.Module):
    def __init__(self, in_channels: int, n_flows: int, split: bool = True, affine: bool = True, conv_lu: bool = True):
        super().__init__()
        self.split = split

        squeeze_dim = 4
        self.flows = nn.ModuleList(
            [Flow(squeeze_dim, affine, conv_lu) for _ in range(n_flows)]
        )

        if self.split:
            # What is this prior...?
            self.prior = ZeroConv2d(in_channels * 2, in_channels * 4)
        else:
            self.prior = ZeroConv2d(in_channels * 4, in_channels * 8)

class Flow(nn.Module):
    def __init__(self, in_channels: int, affine: bool = True, conv_lu: bool = True):
        super().__init__()

        # What is this doing here?
        self.actnorm = ActNorm(in_channels)

        if conv_lu:
            # What is this doing here?
            self.invconv = InvConv2dLU(in_channels)
        else:
            self.invconv = InvConv2d(in_channels)
        
        # What is this doing here?
        self.coupling = AffineCoupling(in_channels, affine)

class ZeroConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding)

        # Why initialise explicitly to zero?
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        # Why create this parameter?
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

class AffineCoupling(nn.Module):
    def __init__(self, in_channels: int, affine: bool = True, filter_size: int = 512):
        super().__init__()

        self.affine = affine

        self.network = nn.Sequential(
            nn.Conv2d(in_channels // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channels if self.affine else in_channels // 2)
        )

class InvConv2dLU(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Go through each component of this class

        weight = np.random.randn(in_channels, in_channels)
        q, _ = linalg.qr(weight)
        w_p, w_l, w_u = linalg.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

class ActNorm(nn.Module):
    def __init__(self, in_channels: int, log_det: bool = True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        self.register_buffer("initialised", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet
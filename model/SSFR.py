import torch
import torch.nn.functional as F
import torch.nn as nn


class SFR(nn.Module):
    def __init__(self,
                 op_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5):
        super(SFR, self).__init__()

        self.gn = nn.GroupNorm(num_groups=group_num, num_channels=op_channels)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # Separate
        gn_x = self.gn(x)  # [64, 128, 13, 9]
        w_gamma = self.gn.weight / sum(self.gn.weight)  # [128]
        # w_gamma = w_gamma.view(B, -1).unsqueeze(2).unsqueeze(3)  # torch.Size([64, 2, 1, 1])
        w_gamma = w_gamma.view(1, -1, 1, 1)
        weigts = self.sigomid(gn_x * w_gamma)

        w1 = torch.where(weigts > self.gate_treshold, torch.ones_like(weigts), weigts)  # informative weight1
        w2 = torch.where(weigts < self.gate_treshold, torch.zeros_like(weigts), weigts)  # non-informative weight2
        x_1 = w1 * x
        x_2 = w2 * x

        # Reconstruct
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CFR(nn.Module):
    def __init__(self,
                 num_channels: int,
                 alpha: float = 0.5,
                 squeeze_radio: int = 2,
                 group_kernel_size: int = 3):
        super(CFR, self).__init__()
        self.up_channel = up_channel = int(alpha * num_channels)
        self.low_channel = low_channel = num_channels - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, num_channels, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=up_channel // squeeze_radio)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, num_channels, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, num_channels - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # Fuse
        s1 = self.gap(Y1).squeeze(-1).squeeze(-1)
        s2 = self.gap(Y2).squeeze(-1).squeeze(-1)
        beta = self.softmax(torch.stack([s1, s2], dim=1))
        out = Y1 * beta[:, 0].unsqueeze(-1).unsqueeze(-1) + Y2 * beta[:, 1].unsqueeze(-1).unsqueeze(-1)

        return out


class SSFR(nn.Module):
    def __init__(self,
                 num_channels: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 0.5,
                 squeeze_radio: int = 2,
                 group_kernel_size: int = 3):
        super(SSFR, self).__init__()
        self.conv1x1_first = nn.Conv2d(num_channels, num_channels, kernel_size=1)

        self.sfr = SFR(num_channels,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.cfr = CFR(num_channels,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_kernel_size=group_kernel_size)

        self.conv1x1_last = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1_first(x)
        out = self.sfr(out)
        #print(out.shape)
        out = self.cfr(out)
        #print(out.shape)
        out = self.conv1x1_last(out)
        out += x
        return out


if __name__ == '__main__':
    x = torch.randn(64, 128, 13, 9)
    model = SSFR(128)
    #print(model)
    print(model(x).shape)
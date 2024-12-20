import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CoordAtt


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def apply_mask(x, mask):
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape
    if (g > 1) and (g != c):
        mask = mask.unsqueeze(1).repeat(1, c // g, 1, 1, 1).transpose(1, 2).reshape(b, c, hw_mask, hw_mask)
    return x * mask


def apply_channel_mask(x, mask):
    b, c, h, w = x.shape
    _, g = mask.shape
    if (g > 1) and (g != c):
        mask = mask.repeat(1,c//g).view(b, c//g, g).transpose(-1,-2).reshape(b,c,1,1)
    else:
        mask = mask.view(b,g,1,1)
    return x * mask


def apply_spatial_mask(x, mask):
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape  # [64, 1, 128, 9]
    # print(mask.shape)
    if (g > 1) and (g != c):
        mask = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,hw_mask,hw_mask)

    return x * mask


class Spatial_Mask(nn.Module):
    def __init__(self, in_channels, mask_channel_group, mask_size, feature_size, training=False, dilate_stride=1):
        super(Spatial_Mask, self).__init__()
        self.mask_channel_group = mask_channel_group
        self.mask_size = mask_size
        self.training = training
        self.attn = CoordAtt(in_channels, in_channels)
        self.conv2 = conv1x1(in_channels, mask_channel_group * 2, bias=True)
        self.conv2_flops_pp = self.conv2.weight.shape[0] * self.conv2.weight.shape[1] + self.conv2.weight.shape[1]
        self.conv2.bias.data[:mask_channel_group] = 5.0
        self.conv2.bias.data[mask_channel_group + 1:] = 1.0
        self.feature_size = feature_size
        self.expandmask = ExpandMask(stride=dilate_stride, mask_channel_group=mask_channel_group)

    def forward(self, x, temperature):
        # attention / dynamic
        # mask = self.attn(x)
        mask = F.adaptive_avg_pool2d(x, self.mask_size) if self.mask_size[0] < x.shape[2] else x
        mask = self.conv2(mask)
        # print(mask.shape)

        # print(mask.shape)
        b, c, h, w = mask.shape
        mask = mask.view(b, 2, c // 2, h, w)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:, 0]
        else:
            mask = (mask[:, 0] >= mask[:, 1]).float()

        if h < self.feature_size[0] and w < self.feature_size[1]:
            mask = F.interpolate(mask, size=(self.feature_size[0], self.feature_size[1]))
        mask_dil = self.expandmask(mask)

        return mask


class Channel_Mask(nn.Module):
    def __init__(self, in_channels, channel_dyn_group, reduction=16):
        super(Channel_Mask, self).__init__()
        self.channel_dyn_group = channel_dyn_group

        self.conv = nn.Sequential(
            conv1x1(in_channels, in_channels // reduction),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(),
        )
        self.linear = nn.Linear(in_channels // reduction, channel_dyn_group * 2, bias=True)

        self.linear.bias.data[:channel_dyn_group] = 2.0
        self.linear.bias.data[channel_dyn_group + 1:] = -2.0

    def forward(self, x, temperature):
        mask = self.conv(x)
        b, c, h, w = mask.shape
        mask = F.adaptive_avg_pool2d(mask, (1, 1)).view(b, c)

        mask = self.linear(mask)

        b, c = mask.shape
        mask = mask.view(b, 2, c // 2)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:, 0]
        else:
            mask = (mask[:, 0] >= mask[:, 1]).float()

        return mask


class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1, mask_channel_group=1):
        super(ExpandMask, self).__init__()
        self.stride = stride
        self.padding = padding
        self.mask_channel_group = mask_channel_group

    def forward(self, x):
        if self.stride > 1:
            self.pad_kernel = torch.zeros((self.mask_channel_group, 1, self.stride, self.stride), device=x.device)
            self.pad_kernel[:, :, 0, 0] = 1
        self.dilate_kernel = torch.ones(
            (self.mask_channel_group, self.mask_channel_group, 1 + 2 * self.padding, 1 + 2 * self.padding),
            device=x.device)

        x = x.float()

        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5


class DSCM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, output_size, stride=1, downsample=None, group_width=1, dilation=1, norm_layer=None,
                 mask_channel_group=1, mask_spatial_granularity=4):
        super(DSCM, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        base_width = 64
        width = int(planes * (base_width / 64.)) * group_width
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, group_width, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.output_size = output_size
        self.mask_spatial_granularity = mask_spatial_granularity
        self.mask_size_h = self.output_size[0] // self.mask_spatial_granularity
        self.mask_size_w = self.output_size[1] // self.mask_spatial_granularity

        self.spatial_mask = Spatial_Mask(inplanes, mask_channel_group, (self.mask_size_h, self.mask_size_w), feature_size=(output_size[0], output_size[1]),
                             dilate_stride=stride)
        self.channel_mask = Channel_Mask(inplanes, mask_channel_group)

    def forward(self, x, temperature=1.0):
        identity = x
        spatial_mask = self.spatial_mask(x, temperature)
        channel_mask = self.channel_mask(x, temperature)
        # print(mask.shape)
        # feature enhancement / reconstruction
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        #
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)

        out1 = apply_spatial_mask(x, spatial_mask)
        out2 = apply_channel_mask(x, channel_mask)
        # print(out1.shape)
        # print(out2.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out1 + out2 + identity
        out = self.relu(out)

        return out





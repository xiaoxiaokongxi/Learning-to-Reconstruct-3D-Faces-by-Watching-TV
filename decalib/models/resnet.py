"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Loads different resnet models
"""
'''
    file:   Resnet.py
    date:   2018_05_02
    author: zhangxiong(1025679612@qq.com)
    mark:   copied from pytorch source code
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import math
import torchvision
import pdb
from .basicblock import PixelShufflePack, ResidualBlockwithBN, ResidualBlockNoBN, make_layer, flow_warp, ResidualBlocksWithInputConv, SecondOrderDeformableAlignment, MSBasicVSRPlusPlus, SPyNet



# TemporalResNet
class TemporalResNet(nn.Module):
    def __init__(self,
                block,
                layers,
                num_classes=1000,
                mid_channels=64,
                num_blocks=5,
                num_res_blocks=5,
                spynet_pretrained='/cluster/work/cvl/jiezcao/pretrained_model/Spynet/spynet_20210409-c6c1bd09.pth',
                max_residue_magnitude=10,
                is_low_res_input=False,
                cpu_cache_length=100,
                ):
        self.inplanes = 64
        super(TemporalResNet, self).__init__()

        self.spynet = SPyNet(pretrained=spynet_pretrained)
        self.spynet.requires_grad_(False)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        self.head = nn.Conv2d(3, mid_channels, 3, 1, 1, bias=True)
        self.down1 = nn.Sequential(make_layer(ResidualBlockwithBN, num_res_blocks, mid_channels=mid_channels), nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True))
        self.rnn = MSBasicVSRPlusPlus(mid_channels, num_blocks, max_residue_magnitude, is_low_res_input, cpu_cache_length, scale_factor=2)
        self.down2 = nn.Sequential(make_layer(ResidualBlockwithBN, num_res_blocks, mid_channels=mid_channels), nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
       # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def flow_forward(self, lqs, scale_factor):
        n, t, c, h, w = lqs.size()

        if scale_factor:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=1./scale_factor,
                mode='bicubic').view(n, t, c, h // scale_factor, w // scale_factor) # TODO

        # compute optical flow using the low-res inputs
        # assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
        #     'The height and width of low-res inputs must be at least 64, '
        #     f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        return flows_forward, flows_backward

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        n, t, c, h, w = x.size()
        # pdb.set_trace()
        flow = self.flow_forward(x, 2)
        x = self.head(x.view(-1, c, h, w))                # [5, 64, 224, 224]
        x = self.down1(x)                                  # [5, 64, 112, 112]
        x = self.rnn(x.view(n, t, -1, h//2, w//2), flow)  # [5, 64, 112, 112]
        # x = self.rnn(x.view(n, t, -1, h, w), flow)
        x = self.down2(x)   # [5, 64, 56, 56]
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        print(x.max(), x.min())
        # x = self.conv1(x.view(-1, c, h, w))  #[10, 64, 112, 112]
        # x = self.bn1(x)      #[10, 64, 112, 112]
        # x = self.relu(x)     #[10, 64, 112, 112]
        # x = self.maxpool(x)  #[10, 64, 56, 56]

        x = self.layer1(x)  #[10, 256, 56, 56]
        x = self.layer2(x)  #[10, 512, 28, 28]
        x = self.layer3(x)  #[10, 1024, 14, 14]
        x1 = self.layer4(x) #[10, 2048, 7, 7]

        x2 = self.avgpool(x1) #[10, 2048, 1, 1]
        x2 = x2.view(x2.size(0), -1)  #[10, 2048]
        # x = self.fc(x)
        ## x2: [bz, 2048] for shape
        ## x1: [bz, 2048, 7, 7] for texture
        return x2


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
       # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)

        x2 = self.avgpool(x1)
        x2 = x2.view(x2.size(0), -1)
        # x = self.fc(x)
        ## x2: [bz, 2048] for shape
        ## x1: [bz, 2048, 7, 7] for texture
        return x2

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def copy_parameter_from_resnet(model, resnet_dict):
    cur_state_dict = model.state_dict()

    for name, param in list(resnet_dict.items())[0:None]:

        if name not in cur_state_dict:
            # print(name, ' not available in reconstructed resnet')
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            cur_state_dict[name].copy_(param)
        except:
            # print(name, ' is inconsistent!')
            continue
    # print('copy resnet state dict finished!')


def load_ResNet50Model():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    copy_parameter_from_resnet(model, torchvision.models.resnet50(pretrained = False).state_dict())
    return model

def load_TemporalResNet50Model():
    model = TemporalResNet(Bottleneck, [3, 4, 6, 3])
    copy_parameter_from_resnet(model, torchvision.models.resnet50(pretrained = False).state_dict())  #TODO
    return model

def load_ResNet101Model():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    copy_parameter_from_resnet(model, torchvision.models.resnet101(pretrained = True).state_dict())
    return model

def load_ResNet152Model():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    copy_parameter_from_resnet(model, torchvision.models.resnet152(pretrained = True).state_dict())
    return model

# model.load_state_dict(checkpoint['model_state_dict'])


######## Unet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

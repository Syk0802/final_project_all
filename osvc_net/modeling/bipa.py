# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import BasicBlock, Bottleneck, ResNet
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.senet import (
    SEBottleneck,
    SENet,
    SEResNetBottleneck,
    SEResNeXtBottleneck,
)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BiPA(nn.Module):
    in_planes = 2048

    def __init__(
        self,
        num_classes,
        last_stride,
        model_path,
        neck,
        neck_feat,
        model_name,
        pretrain_choice,
    ):
        super(BiPA, self).__init__()
        if model_name == "resnet18":
            self.in_planes = 512
            self.base = ResNet(
                last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2]
            )
        elif model_name == "resnet34":
            self.in_planes = 512
            self.base = ResNet(
                last_stride=last_stride, block=BasicBlock, layers=[3, 4, 6, 3]
            )
        elif model_name == "resnet50":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3]
            )
        elif model_name == "resnet101":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 23, 3]
            )
        elif model_name == "resnet152":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 8, 36, 3]
            )

        elif model_name == "se_resnet50":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 6, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet101":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 23, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet152":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 8, 36, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext50":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 6, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext101":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 23, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "senet154":
            self.base = SENet(
                block=SEBottleneck,
                layers=[3, 8, 36, 3],
                groups=64,
                reduction=16,
                dropout_p=0.2,
                last_stride=last_stride,
            )
        elif model_name == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......")

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.bottleneck1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.bottleneck2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)
        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)

    def forward(self, x, label=None, vis=False):
        x = self.base(x)
        x41, x42 = torch.chunk(x, 2, 2)
        x41 = self.gap(x41)
        x42 = self.gap(x42)

        x41 = x41.view(x41.size(0), x41.size(1))
        x42 = x42.view(x42.size(0), x42.size(1))

        feat41 = self.bottleneck1(x41)
        feat42 = self.bottleneck2(x42)

        if self.training:
            return self.classifier1(feat41), self.classifier2(feat42), x41, x42
        else:

            if not vis:
                return torch.cat((feat41, feat42),1)
            else:
                localization_map_normed = self.cam(label, x)
                return localization_map_normed


    def load_param(self, trained_path, skip_cls=True):
        param_dict = torch.load(trained_path).state_dict()  # by wyf
        for i in param_dict:
            if "classifier" in i and skip_cls:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def cam(self, label, feature):
        label = torch.LongTensor(label)
        weight1 = self.classifier1.weight[label][..., None, None]
        weight2 = self.classifier2.weight[label][..., None, None]
        x41, x42 = torch.chunk(feature, 2, 2)

        x41 = x41 * weight1
        x42 = x42 * weight2
        x41 = x41.mean(dim=1)
        x42 = x42.mean(dim=1)
        cam = torch.cat((x41, x42), dim=-2)

        return cam

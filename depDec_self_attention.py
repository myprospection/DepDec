import torch
from torch import nn
from torch.nn import functional as F
import resnet
from utils1 import IntermediateLayerGetter
from utils1 import _SimpleSegmentationModel
import numpy as np

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[6, 12, 18]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.aspp = ASPP(in_channels,  aspp_dilate)
        self.image_classifier = nn.Sequential(
            # 步骤1：初步特征提取
            nn.Conv2d(304, 256, 3, padding=1, stride=2),  # (4,64,20,30)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 步骤2：特征深化
            nn.Conv2d(256, 128, 3, padding=1, stride=2),  # (4,128,10,15)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 步骤3：多尺度特征融合
            nn.Conv2d(128, 64, 3, padding=1, stride=1),  # (4,256,10,15)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 步骤4：全局特征压缩
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # 将特征图展平
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        # 不就是这个么  一样的，没什么多大的区别， 我多加了个resnet34的残差而已
        self.pixel_classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 304 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, bias=False),  # 256 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        # self.pixel_classifier = PixelClassifier(num_classes)

    def forward(self, feature):
        low_level_feature = self.project(
            feature['low_level'])  # return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        aspp_fearture = self.aspp(feature['out'])
        output_new_feature = F.interpolate(aspp_fearture, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        pixel_logits = self.pixel_classifier(torch.cat([low_level_feature.clone(), output_new_feature.clone()], dim=1))
        image_out = self.image_classifier(torch.cat([low_level_feature.clone(), output_new_feature.clone()], dim=1))
        return pixel_logits, image_out

    def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0.0001)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.query = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.key = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.value = nn.Conv2d(out_channels, out_channels, 1)
        # self.sqrt_dk = np.sqrt(out_channels // 8)  # 初始化时计算
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.channel_proj(x)
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out * x  # 残差连接


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=None, is_pooling=False):
        super().__init__()
        if is_pooling:
            self.conv = ASPPPooling(in_channels, out_channels)
        else:
            if dilation is None:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False)
                )
            else:
                self.conv = ASPPConv(in_channels, out_channels, dilation)
        self.attention = SelfAttention(in_channels, out_channels)

    def forward(self, x):
        conv_out = self.conv(x)
        attn_out = self.attention(x)
        # gamma = self.gamma
        return conv_out * attn_out  # 残差连接


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256

        modules = []
        # 1x1卷积
        modules.append(ASPPModule(in_channels, out_channels, dilation=None))
        # 3个不同膨胀率的3x3卷积
        for rate in atrous_rates:
            modules.append(ASPPModule(in_channels, out_channels, dilation=rate))
        # 全局池化
        modules.append(ASPPModule(in_channels, out_channels, is_pooling=True))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3(_SimpleSegmentationModel):
    pass


def deeplabv3plus_resnet(num_classes):
    backbone_name='resnet101'
    output_stride= 16
    pretrained_backbone=False
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}  #
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    # 提取网络的第几层输出结果并给一个别名
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


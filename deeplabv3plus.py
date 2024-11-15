import torch
from torch import nn
from torch.nn import functional as F
import resnet
from utils import IntermediateLayerGetter
from utils import _SimpleSegmentationModel
import numpy as np

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        # print(feature.shape)
        low_level_feature = self.project(
            feature['low_level'])  # return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        # print(low_level_feature.shape)
        output_feature = self.aspp(feature['out'])
        # print(output_feature.shape)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        # print(output_feature.shape)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class self_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self_attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = 30
        self.dim_k = 30
        self.Q = nn.Conv2d(out_channels, out_channels // 8, kernel_size=1)
        self.K = nn.Conv2d(out_channels, out_channels // 8, kernel_size=1)
        self.V = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

        # self.bn_query = nn.BatchNorm2d(out_channels // 8)
        # self.bn_key = nn.BatchNorm2d(out_channels // 8)
        self.bn_value = nn.BatchNorm2d(out_channels)
    def forward(self, input):
        batch_size = input.size(0)
        height = input.size(2)
        width = input.size(3)
        Q = self.Q(input).view(batch_size,-1,height*width)
        K = self.K(input).view(batch_size,-1,height*width)
        V = self.V(input).view(batch_size,-1,height*width)
        # print("Q shape:", Q.shape)
        # print("K shape:", K.shape)
        # print("V shape:", V.shape)
        # attention_weights = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.dim_k)
        attention_weights = F.softmax(torch.bmm(Q.permute(0, 2, 1), K), dim=-1)
        output = torch.bmm(V, attention_weights).view(batch_size, -1, height, width)
        output = self.bn_value(output)
        # attention_weights = F.relu(attention_weights)

        # output =torch.matmul(attention_weights, V)
        # output = self.BatchNorm2d(output),
        output = input + output * 0.00001 # 残差连接
        return output

# # 示例输入
# batch_size = 32
# seq_length = 10
# embedding_dim = 64
# dim_k = 64
#
# inputs = np.random.randn(batch_size, seq_length, embedding_dim)
# output = self_attention(inputs, dim_k)
# print(output.shape)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        self_attention(in_channels, out_channels),
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            #print(conv(x).shape)
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
#resnet34 把向量压缩成1维的然后输出分类结果
class DeepLabV3(_SimpleSegmentationModel):
    pass

def deeplabv3plus_resnet(num_classes):
    backbone_name='resnet101'
    output_stride=8
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

def correction_resnet(num_classes):
    backbone_name = 'resnet101'
    output_stride = 8
    pretrained_backbone = False
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,input=num_classes+2)

    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}  #
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    # 提取网络的第几层输出结果并给一个别名
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model
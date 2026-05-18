import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from torch import Tensor
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict

import torch
import torch.nn.functional as F
import math
from torch import nn

from torch.nn.parameter import Parameter

class _DenseLayer(nn.Module):
    def __init__(self,
                 input_c: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        super(_DenseLayer, self).__init__()
        # 第一个卷积层
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        self.add_module("selayer", GCT(input_c))
        self.add_module("norm1", nn.BatchNorm2d(input_c))  # 添加一个BatchNorm2d层，用于对输入进行批标准
        self.add_module("relu1", nn.LeakyReLU(inplace=True))  # 添加一个ReLU激活函数层，将其应用在输入张量上
        self.add_module("conv1", nn.Conv2d(in_channels=input_c,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))  # 添加一个1x1卷积层

        # 第二个卷积层
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.LeakyReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))  # 添加一个3x3卷积层

        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        # 将输入张量连接在通道维度上
        concat_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        # 检查输入张量是否有梯度要求
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        # 使用checkpointing执行forward函数中的一部分，以节省内存
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")
            # 使用checkpointing执行bn_function函数
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,
                 num_layers: int,
                 input_c: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            # 创建_DenseLayer层，并添加到模型中
            layer = _DenseLayer(input_c + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:

        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    # 使用1*1卷积和2*2平均池化作为两个连续密集块之间的过渡层
    def __init__(self,
                 input_c: int,
                 output_c: int):
        super(_Transition, self).__init__()
        # 过渡层
        self.add_module("norm", nn.BatchNorm2d(input_c))  # 对输入进行通道维度上的标准化操
        self.add_module("relu", nn.LeakyReLU(inplace=True))  # 添加一个ReLU激活函数层
        self.add_module("conv", nn.Conv2d(input_c,
                                          output_c,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))  # 添加一个1x1卷积层，用于调整通道数GCT
        self.add_module("pool", SoftPool2D(kernel_size=2, stride=2))  # ：添加一个平均池化层，将特征图的尺寸减少一半


class DenseNet(nn.Module):
    """
growth_rate（增长率）是指每个层中要添加的滤波器（即卷积核）的数量。它决定了每个密集块（dense block）中每个层的输出通道数。
block_config（块配置）是一个包含4个整数的列表，用于指定每个池化块（pooling block）中有多少层。例如，(6, 12, 24, 16) 表示第一个池化块有6层，第二个池化块有12层，以此类推。
num_init_features（初始特征数）是指在第一个卷积层中学习的滤波器（卷积核）的数量。它决定了输入图像经过第一个卷积层后的输出通道数。
bn_size（瓶颈层倍数）是一个乘性因子，用于确定瓶颈层中的特征映射通道数。即瓶颈层的输出通道数为 bn_size * growth_rate。
drop_rate（丢弃率）是在每个密集层（dense layer）后应用的丢弃（dropout）比率。丢弃是一种正则化技术，用于减少过拟合。
num_classes（分类类别数）是要分类的类别数量。这决定了最终全连接层的输出维度，与数据集的类别数相匹配。
memory_efficient（内存效率）是一个布尔值，表示是否使用内存效率的检查点（checkpointing）技术。当设置为 True 时，模型使用检查点技术以节省内存，但会导致计算效率稍微降低。当设置为 False 时，不使用检查点技术。
    """

    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 48, 32),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False):
        super(DenseNet, self).__init__()
        # 第一层卷积conv+bn+relu+pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.LeakyReLU(inplace=True)),
            ("pool0", SoftPool2D(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add SELayer at first convolution
        # self.features.add_module("SELayer_0a", SELayer(channel=num_init_features))

        # 每个Dense Block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            # Add a SELayer
            self.features.add_module("GCTLayer_%da" % (i + 1), GCT(num_features))
            # 创建_DenseBlock层，并添加到模型
            block = _DenseBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                # 创建_Transition层，并添加到模型中
                # Add a SELayer behind each transition block
                self.features.add_module("GCTLayer_%db" % (i + 1), GCT(num_features))
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

            # 最后的Batch Normalization层
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        # 全连接层
        self.classifier = nn.Linear(num_features, num_classes)


        # 初始化模型中的权重和偏置
        for m in self.modules():
            # 卷积层使用 Kaiming 正态分布初始化方法，适用于激活函数为 ReLU
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# DenseNet121(k=32):blocks=[6,12,24,16]
# DenseNet169(k=32):blocks=[6,12,32,32]
# DenseNet201(k=32):blocks=[6,12,48,32]
# DenseNet161(k=48):blocks=[6,12,36,24]
model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}
def densenet121(**kwargs: Any) -> DenseNet:
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    model = DenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=64,
                    **kwargs)
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = model_zoo.load_url(model_urls['densenet121'])
    num_classes = model.classifier.out_features
    load_fc = num_classes == 2

    for key in list(state_dict.keys()):
        if load_fc is False:
            if "classifier" in key:
                del state_dict[key]

        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=load_fc)

    print("successfully load pretrain-weights.")
    return model


def densenet169(**kwargs: Any) -> DenseNet:
    # Top-1 error: 24.00%
    # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32, 32),
                    num_init_features=64,
                    **kwargs)


def densenet201(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.80%
    # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 48, 32),
                    num_init_features=64,
                    **kwargs)


def densenet161(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.35%
    # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
    return DenseNet(growth_rate=48,
                    block_config=(6, 12, 36, 24),
                    num_init_features=96,
                    **kwargs)


def load_state_dict(model: nn.Module, weights_path: str) -> None:
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(weights_path)

    num_classes = model.classifier.out_features
    load_fc = num_classes == 1000

    for key in list(state_dict.keys()):
        if load_fc is False:
            if "classifier" in key:
                del state_dict[key]

        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=load_fc)

    print("successfully load pretrain-weights.")
# se_densenet = DenseNet()
# print('==================1. 通过print打印网络结构=====================')
# print(se_densenet)   # 1. 通过print打印网络结构
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())
#
# 创建一个DenseNet模型
# model = densenet121()
#
# # 计算参数数量
# num_params = count_parameters(model)
# print("Total number of parameters: {}".format(num_params))


class GhostSEModule(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=1, ratio=2, dw_size=3):
        super(GhostSEModule, self).__init__()
        self.ghost_module = GhostModule(channel, channel, kernel_size, ratio, dw_size)
        self.se_layer = SELayer(channel, reduction)

    def forward(self, x):
        x = self.ghost_module(x)
        x = self.se_layer(x)
        return x


class GhostECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2, ratio=2, dw_size=3):
        super(GhostECA, self).__init__()
        # 根据通道数求出卷积核的大小kernel_size
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ghost_module = GhostModule(channel, channel, kernel_size=kernel_size, ratio=ratio, dw_size=dw_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ghost_module(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# class GhostECA(nn.Module):
#     def __init__(self, channel, b=1, gamma=2, ratio=2):
#         super(GhostECA, self).__init__()
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.ghost = GhostModule(channel, channel // ratio)
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)  # 通道权重计算
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.ghost(y)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)


# class GhostModule(nn.Module):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         # ratio一般会指定成2，保证输出特征层的通道数等于exp
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels * (ratio - 1)
#
#         # 利用1x1卷积对输入进来的特征图进行通道的浓缩，获得特征通缩
#         # 跨通道的特征提取
#         self.primary_conv = nn.Sequential(
#             nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
#             # 1x1卷积的输入通道数为GhostModule的输出通道数oup/2
#             nn.BatchNorm2d(init_channels),  # 1x1卷积后进行标准化
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),  # ReLU激活函数
#         )
#
#         # 在获得特征浓缩后，使用逐层卷积，获得额外的特征图
#         # 跨特征点的特征提取    一般会设定大于1的卷积核大小
#         self.cheap_operation = nn.Sequential(
#             nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
#             # groups参数的功能就是将普通卷积转换成逐层卷据
#             nn.BatchNorm2d(new_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )
#
#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         # 将1x1卷积后的结果和逐层卷积后的结果进行堆叠
#         out = torch.cat([x1, x2], dim=1)
#         return out[:, :self.oup, :, :]


class GhostModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, ratio=2, dw_size=3, relu=True):
        super(GhostModule, self).__init__()
        self.output_channels = output_channels  # 设定输出通道数，即最终的通道数
        init_channels = output_channels // ratio  # 计算Ghost模块中主干部分的通道数
        new_channels = init_channels * (ratio - 1)  # 计算Ghost模块中影子部分的通道数

        # 主干部分，进行普通的卷积操作
        self.primary_conv = nn.Sequential(
            nn.Conv2d(input_channels, init_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()  # 是否使用ReLU激活函数取决于relu参数
        )

        # 影子部分，进行depthwise卷积操作
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, stride=1, padding=(dw_size - 1) // 2, groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()  # 是否使用ReLU激活函数取决于relu参数
        )

    def forward(self, x):
        x1 = self.primary_conv(x)  # 主干部分的卷积操作，得到主干部分的输出
        x2 = self.cheap_operation(x1)  # 影子部分的depthwise卷积操作，得到影子部分的输出
        out = torch.cat([x1, x2], dim=1)  # 沿着通道维度进行拼接，将主干部分和影子部分合并
        return out[:, :self.output_channels, :, :]  # 返回合并后的张量，并截取前面设定的输出通道数


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        assert channel > reduction, "Make sure your input channel bigger than reduction which equals to {}".format(
            reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        # ----------------------------------#
        # 根据通道数求出卷积核的大小kernel_size
        # ----------------------------------#
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------#
        # 显示全局平均池化,再是k*k的卷积,
        # 最后为Sigmoid激活函数,进而得到每个通道的权重w
        # 最后进行回承操作,得出最终结果
        # ------------------------------------------#
        y = self.avg_pool(x)
        # y.squeeze(-1)是将最后一个维度删掉即宽这个维度就没有了，transpose(-1, -2)是将最后一个和倒数第二个维度进行互换，即现在的维度变成了b，1，c这三个维度，1是由于前面的自适应平均层变成了1*1的图像，所以长在这里就是1。unsqueeze(-1)是增加最后一个维度
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g


class GELayer(nn.Module):
    def __init__(self, channel, layer_idx):
        super(GELayer, self).__init__()

        # Kernel size w.r.t each layer for global depth-wise convolution
        kernel_size = [-1, 56, 28, 14, 7][layer_idx]

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel_size, groups=channel),
            nn.BatchNorm2d(channel),
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        gate = self.conv(x)
        gate = self.activation(gate)
        return x * gate


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None, force_inplace=False):
        kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = (stride, stride)
        _, c, h, w = x.shape
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(
            F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))

#
def soft_pool2d(x, kernel_size=2, stride=None):
    kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = (stride, stride)
    _, c, h, w = x.shape
    e_x = torch.exp(x)
    return F.avg_pool2d(x * e_x, kernel_size, stride=stride) * (sum(kernel_size)) / (
            F.avg_pool2d(e_x, kernel_size, stride=stride) * (sum(kernel_size)))


class SoftPool2D(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(SoftPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x


class mixedPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, alpha=0.5):
        # nn.Module.__init__(self)
        super(mixedPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)  # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (
                1 - self.alpha) * F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x

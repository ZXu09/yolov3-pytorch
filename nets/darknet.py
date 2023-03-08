import math
from collections import OrderedDict

import torch.nn as nn


# ---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
# ---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    # 类中的函数称为方法，类定义了__init__()方法，类的实例化操作会自动调用__init__方法。
    def __init__(self, inplanes, planes):  # 初始化实例后的对象，self代表的是类的实例（对象）
        # 一般神经网络的类都继承自 torch.nn.Module
        # init() 和 forward() 是自定义类的两个主要函数，在自定义类的 init() 中需要添加一句 super(Net, self).init()，其中 Net 是自定义的类名，用于继承父类的初始化函数。
        # 注意在 init() 中只是对神经网络的模块进行了声明，真正的搭建是在 forward() 中实现。自定义类的成员都通过 self 指针来访问，所以参数列表中都包含了 self
        super(BasicBlock, self).__init__()
        # 1×1卷积下降通道数
        self.conv1 = nn.Conv2d(
            inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        # BN的目的是使一批(Batch)feature map满足均值为0，方差为1的分布（均值归一化）
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        # 3×3卷积扩张通道数
        self.conv2 = nn.Conv2d(
            planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

        # 这样可以减少参数，参数量计算：channel_in * kernel_size × kernel_size *channel_out
    def forward(self, x):
        residual = x  # 残差边（直接映射）

        out = self.conv1(x)  # 主干边（残差块）
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        # nn.Conv2d是二维卷积方法，输入通道数3，输出通道数32，卷积核大小3×3，步长为1，图像四周填充0
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)  # 标准化
        # torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        # negative_slope：控制负斜率的角度，默认等于0.01
        # inplace-选择是否进行覆盖运算
        # 1.能解决深度神经网络（层数非常多）的"梯度消失"问题（线性的不值域饱和，非线性的复杂映射，在小于0的时候梯度为0，大于0的时候梯度恒为1），浅层神经网络（三五层那种）才用sigmoid 作为激活函数。
        # 2.它能加快收敛速度。
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        # planes[0] = 32, planes[1] = 64
        # 1、3×3卷积416,416,32 -> 208,208,64
        # 2、1×1卷积降维208,208,64->208,208,32
        # 3、3×3卷积升维208,208,32->208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    # ---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):  # blocks表示循环次数
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(
            self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(
                i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):  # 前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])  # 列表，对应残差块的重复次数
    return model

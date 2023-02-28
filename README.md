# yolov3-pytorch
## 一、预测部分
主干特征提取网络darknet53介绍
输入416×416×3->进行下采样，宽高不断压缩，通道数不断扩张；若是进行上采样，宽高不断扩张，通道数不断压缩。
![darknet53](https://github.com/SZUZOUXu/yolov3-pytorch/blob/main/image/darknet53.jpg)

**残差网络Residual**->Deep-Learning/Resnet50；h(x)=F(x)（残差部分，包含卷积）+x(直接映射)
1. Darknet53具有一个重要特点是使用了**残差网络Residual**，Darknet53中的残差卷积就是首先进行一次卷积核大小为3X3、步长为2的卷积，
该卷积会压缩输入进来的特征层的宽和高，此时我们可以获得一个特征层，我们将该特征层命名为layer。之后我们再对该特征层进行一次1X1的卷积和一次3X3的卷积，
并把这个结果加上layer，此时我们便构成了残差结构。

通过不断的1X1卷积和3X3卷积以及残差边的叠加，我们便大幅度的加深了网络。
**残差网络**的特点是容易优化，并且能够通过增加相当的深度来提高准确率。其内部的残差块使用了**跳跃连接**，缓解了在深度神经网络中增加深度带来的**梯度消失**问题。

2. Darknet53的每一个卷积部分使用了特有的DarknetConv2D结构，每一次卷积的时候进行l2正则化，完成卷积后进行**BatchNormalization标准化与LeakyReLU**。
普通的ReLU是将所有的负值都设为零，**Leaky ReLU**则是给所有**负值赋予一个非零斜率**。以数学的方式我们可以表示为：
![LeakyReLU](https://github.com/SZUZOUXu/yolov3-pytorch/blob/main/image/LeakyReLU.png)

代码如下：
```Python
import math
from collections import OrderedDict

import torch.nn as nn


#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    # 类中的函数称为方法，类定义了__init__()方法，类的实例化操作会自动调用__init__方法。
    def __init__(self, inplanes, planes):# 初始化实例后的对象，self代表的是类的实例（对象）
        # 一般神经网络的类都继承自 torch.nn.Module
        # init() 和 forward() 是自定义类的两个主要函数，在自定义类的 init() 中需要添加一句 super(Net, self).init()，其中 Net 是自定义的类名，用于继承父类的初始化函数。
        # 注意在 init() 中只是对神经网络的模块进行了声明，真正的搭建是在 forward() 中实现。自定义类的成员都通过 self 指针来访问，所以参数列表中都包含了 self
        super(BasicBlock, self).__init__()
        # 1×1卷积下降通道数
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])# BN的目的是使一批(Batch)feature map满足均值为0，方差为1的分布（均值归一化）
        self.relu1  = nn.LeakyReLU(0.1)
        
        #3×3卷积扩张通道数
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)
        
        #这样可以减少参数，参数量计算：channel_in * kernel_size × kernel_size *channel_out
    def forward(self, x):
        residual = x# 残差边（直接映射）

        out = self.conv1(x)# 主干边（残差块）
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
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)# 标准化
        # torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        # negative_slope：控制负斜率的角度，默认等于0.01
        # inplace-选择是否进行覆盖运算
        # 1.能解决深度神经网络（层数非常多）的"梯度消失"问题（线性的不值域饱和，非线性的复杂映射，在小于0的时候梯度为0，大于0的时候梯度恒为1），浅层神经网络（三五层那种）才用sigmoid 作为激活函数。
        # 2.它能加快收敛速度。
        self.relu1  = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
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

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):# blocks表示循环次数
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):# 前向传播
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
    model = DarkNet([1, 2, 8, 8, 4])# 列表，对应残差块的重复次数
    return model

```

## 二、从特征获取预测结果
从特征获取预测结果的过程可以分为两个部分，分别是：

- 构建**FPN特征金字塔**进行加强**特征提取**。
- 利用**Yolo Head**对三个有效特征层进行**预测**。
### 构建FPN特征金字塔进行加强特征提取
在特征利用部分，YoloV3提取多特征层进行目标检测，一共提取**三个特征层**。
三个特征层位于主干部分Darknet53的不同位置，分别位于中间层，中下层，底层，三个特征层的shape分别为(52,52,256)、(26,26,512)、(13,13,1024)。

在获得三个有效特征层后，我们利用这三个有效特征层进行**FPN层的构建**，构建方式为：

1. 13x13x1024的特征层进行5次卷积处理，处理完后利用**YoloHead**获得预测结果
2. 一部分用于进行**上采样UmSampling2**d后与26x26x512特征层进行结合，结合特征层的shape为(26,26,768)。结合特征层再次进行5次卷积处理，处理完后利用**YoloHead**获得预测结果.
3. 一部分用于进行**上采样UmSampling2d**后与52x52x256特征层进行结合，结合特征层的shape为(52,52,384)。结合特征层再次进行5次卷积处理，处理完后利用**YoloHead**获得预测结果。
**特征金字塔**可以将**不同shape的特征层**进行**特征融合**，有利于提取出更好的特征。

### 利用Yolo Head获得预测结果
利用**FPN特征金字塔**，我们可以获得**三个加强特征**，这三个加强特征的shape分别为(13,13,512)、(26,26,256)、(52,52,128)，然后我们利用这三个shape的特征层传入Yolo Head获得预测结果。

Yolo Head本质上是一次3x3卷积加上一次1x1卷积，**3x3卷积的作用是特征整合，1x1卷积的作用是调整通道数**。

对三个特征层分别进行处理，假设我们预测是的VOC数据集，我们的输出层的shape分别为(13,13,75)，(26,26,75)，(52,52,75)

(13,13,75)->(13,13,3,25)13×13的网格，每个网格3个先验框（预先标注在图片上，预测结果判断先验框内部有无物体，对先验框进行中心还有宽高的调整）->(13,13,3,20+1+4)VOC数据集分20个类，1判断先验框内是否有物体，4代表先验框的调整参数（4个参数确定一个框的位置）最后的维度应该为**75 = 3x25**

如果使用的是coco训练集，类则为80种，最后的维度应该为**255 = 3x85**，三个特征层的shape为(13,13,255)，(26,26,255)，(52,52,255)

其实际情况就是，输入N张416x416的图片，在经过多层的运算后，会输出三个shape分别为(N,13,13,255)，(N,26,26,255)，(N,52,52,255)的数据，对应每个图分为13x13、26x26、52x52的网格上3个先验框的位置。

代码如下：
```Python
from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),# 1×1的卷积调整通道数
        conv2d(filters_list[0], filters_list[1], 3),# 3×3的卷积进行特征提取
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),# 后面这两层实现分类预测和回归预测
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        # 3 * (4 + 1 + 20)
        # (13,13,512)，输出的out_filter对应75，out_filters[-1]表示最后一层的输入，中间的层之间的变换为[512,1024]
        # out_filters[-1]->512->1024...->1024->75
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        # 卷积 + 上采样
        self.last_layer1_conv       = conv2d(512, 256, 1) # 1×1的卷积实现通道减半， in_channels, out_channels, kernel_size:
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')# 上采样，放大倍数为2
        # (26,26,256)，高、宽26
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        
        # 卷积 + 上采样
        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
         # (52,52,128)，高、宽26
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层x2,x1,x0，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2

```

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
普通的ReLU是将所有的负值都设为零，**Leaky ReLU**则是给所有负值赋予一个非零斜率。以数学的方式我们可以表示为：
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
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])# BN的目的是使一批(Batch)feature map满足均值为0，方差为1的分布（均值归一化）
        self.relu1  = nn.LeakyReLU(0.1)
        
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
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
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)# nn.Conv2d()类作为二维卷积的实现
        self.bn1    = nn.BatchNorm2d(self.inplanes)
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
    def _make_layer(self, planes, blocks):
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

    def forward(self, x):
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
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
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
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
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

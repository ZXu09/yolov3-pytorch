- [yolov3-pytorch](#yolov3-pytorch)
- [一、预测部分](#一预测部分)
  - [1.主干特征提取网络darknet53介绍](#1主干特征提取网络darknet53介绍)
  - [二、从特征获取预测结果](#二从特征获取预测结果)
    - [构建FPN特征金字塔进行加强特征提取](#构建fpn特征金字塔进行加强特征提取)
    - [利用Yolo Head获得预测结果](#利用yolo-head获得预测结果)
  - [3.预测结果的解码](#3预测结果的解码)
  - [4.在原图上进行绘制](#4在原图上进行绘制)
- [二、训练部分](#二训练部分)

生成目录：快捷键CTRL(CMD)+SHIFT+P，输入Markdown All in One: Create Table of Contents回车
# yolov3-pytorch
# 一、预测部分
## 1、主干特征提取网络darknet53介绍
输入416×416×3->进行下采样，宽高不断压缩，同时通道数不断扩张；若是进行上采样，宽高不断扩张，同时通道数不断压缩。
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

## 2、从特征获取预测结果
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
         # (52,52,128)，高、宽52
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
        out0_branch = self.last_layer0[:5](x0)# out0_branch (13,13,512)然后送去1×1降维、上采样
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)# x1_in (26,26,256)送去叠加
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)# x1_in (26,26,768)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in) # out1_branch (26,26,256)然后送去1×1降维、上采样
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)# x2_in(52,52,128)送去叠加
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)#  x2_in (52,52,384)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2# 预测结果，对先验框进行调整

```
## 3、预测结果的解码
由第二步我们可以获得三个特征层的预测结果，shape分别为：

- (N,13,13,255)/(N,13,13,75)
- (N,26,26,255)/(N,13,13,75)
- (N,52,52,255)/(N,13,13,75)

在这里我们简单了解一下每个有效特征层到底做了什么：

每一个有效特征层将整个图片分成与其长宽对应的网格，如(N,13,13,255)/(N,13,13,75)的特征层就是将整个图像分成**13x13个网格**；然后从**每个网格中心建立多个先验框**，这些框是网络预先设定好的框，网络的预测结果会**判断这些框内是否包含物体，以及这个物体的种类**。

由于每一个网格点都具有三个先验框，所以上述的预测结果可以reshape为：

- (N,13,13,3,85)/(N,13,13,3,25)
- (N,26,26,3,85)/(N,13,13,3,25)
- (N,52,52,3,85)/(N,13,13,3,25)
其中的85可以拆分为4+1+80，25可以拆分为4+1+20其中的4代表先验框的调整参数，1代表先验框内是否包含物体，80代表的是这个先验框的种类，由于coco分了80类，所以这里是80。

4+1对应：**x_offset、y_offset、h和w、置信度**。

YoloV3的解码过程分为两步：

- 先将**每个网格点加上它对应的x_offset和y_offset**，加完后的结果就是**预测框的中心**。
- 然后再利用**先验框和h、w结合计算出预测框的宽高**。这样就能得到整个预测框的位置了。


得到最终的预测结果后还要进行**得分排序**与**非极大抑制筛选**。

这一部分基本上是所有目标检测通用的部分。其对于每一个类进行判别：
1. 取出每一类**得分大于self.obj_threshold的框和得分**。
2. 利用**框的位置和得分**进行**非极大抑制**。
代码如下：
```Python
import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors_mask   = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            #-----------------------------------------------#
            batch_size      = input.size(0)# 一共多少图片
            input_height    = input.size(2)
            input_width     = input.size(3)

            #-----------------------------------------------#
            #   输入为416x416时
            #   stride_h = stride_w = 32、16、8
            #   假设为input_height = input_width = 13时，一个特征点对应416/13=32个像素点
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            #-------------------------------------------------#
            #   此时获得的scaled_anchors(先验框)大小是相对于特征层的
            #   计算出先验框在特征层上对应的宽高
            #-------------------------------------------------#
            # 除上32对应13×13的特征层的大小（对于每一个anchors）
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #   batch_size, 3, 52, 52, 85
            #-----------------------------------------------#
            # batch_size,3*(5+num_classes),13,13->batch_size,3,13,13,(5+num_classes)
            # view变换维度，把原先tensor中的数据按行优先的顺序排成一个一维数据，然后按照输入参数要求，组合成其他维度的tensor。
            # permute将tensor中任意维度利用索引调换。torch.Size([2, 2, 3])，b=a.permute(2,0,1)，torch.Size([2, 3, 2])
            # 仅理解contiguous()函数，可以理解为一次深拷贝
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            #-----------------------------------------------#
            #   先验框的中心位置的调整参数
            #-----------------------------------------------#
            # 在0-1之间，故先验框只能在右下角的框内进行调整，让物体由左上角的网格点预测
            x = torch.sigmoid(prediction[..., 0]) #  python切片操作中，...用于代替多个维度，等价于[:, :, 0]，表示最后一个维度的第0列
            y = torch.sigmoid(prediction[..., 1])
            #-----------------------------------------------#
            #   先验框的宽高调整参数
            #-----------------------------------------------#
            w = prediction[..., 2]
            h = prediction[..., 3]
            #-----------------------------------------------#
            #   获得置信度，是否有物体
            #-----------------------------------------------#
            conf        = torch.sigmoid(prediction[..., 4])
            #-----------------------------------------------#
            #   种类置信度
            #-----------------------------------------------#
            pred_cls    = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角（点上） 
            #   batch_size,3,13,13 13×13的网格点上，每个网格点上有3个先验框
            #----------------------------------------------------------#
            # torch.linspace(start,end,steps)返回一个一维的tensor（张量），这个张量包含了从start到end（包括端点）的等距的steps个数据点。
            # input_height、input_width 13×13
            # x.repeat(a,b) :将张量x在列的方向上重复b次，在行的方向上重复a次。->列方向重复input_height 13次->(13×13)
            # x.repeat(a,b,c) :将张量x在列的方向上重复c次，在行的方向上重复b次，在深度的方向上重复a次。->深度方向重复3(每个网格点对应3个先验框)*样本数量->(batch_size*3,13,13)
            # 对已知的进行reshape，并改变数据类型
            # x.shape->(batch_size,3,13,13)
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            #----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            # w.shape (batch_size*3,13,13)
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            #----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #----------------------------------------------------------#
            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x.data + grid_x
            pred_boxes[..., 1]  = y.data + grid_y
            pred_boxes[..., 2]  = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3]  = torch.exp(h.data) * anchor_h

            #----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            #   将输出调整为416*416的形式
            #----------------------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            #----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            #----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            #-------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            #-------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            #------------------------------------------#
            #   获得预测结果中包含的所有种类
            #------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]
                
                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

```
## 4.在原图上进行绘制
通过第三步，我们可以获得预测框在原图上的位置，而且这些预测框都是经过筛选的。这些筛选后的框可以直接绘制在图片上，就可以获得结果了。
# 二、训练部分
## 1、计算loss所需参数
在计算loss的时候，实际上是**pred和target**之间的对比：
- pred就是网络的预测结果。
- target就是网络的真实框情况。

## 2、pred是什么
对于yolo3的模型来说，网络最后输出的内容就是**三个特征层每个网格点对应的预测框及其种类**，即三个特征层分别对应着图片被分为不同size的网格后，**每个网格点上三个先验框对应的位置、置信度及其种类**。

输出层的shape分别为(13,13,75)，(26,26,75)，(52,52,75)，最后一个维度为75是因为是基于voc数据集的，它的类为20种，yolo3只有针对每一个特征层存在3个先验框，所以最后维度为3x25；
如果使用的是coco训练集，类则为80种，最后的维度应该为255 = 3x85，三个特征层的shape为(13,13,255)，(26,26,255)，(52,52,255)

现在的y_pre还是没有解码的，解码了之后才是真实图像上的情况。

## 3、target是什么。
target就是**一个真实图像中，真实框的情况**。
第一个维度是batch_size，第二个维度是每一张图片里面真实框的**数量**，第三个维度内部是真实框的信息，包括**位置以及种类**。

## 4、loss的计算过程
拿到pred和target后，不可以简单的减一下作为对比，需要进行如下步骤。

判断真实框在图片中的位置，判断其**属于哪一个网格点去检测**。判断真实框和这个特征点的**哪个先验框重合程度最高**。计算该网格点应该有**怎么样的预测结果才能获得真实框**，与真实框重合度最高的**先验框被用于作为正样本**。

根据网络的预测结果获得**预测框**，计算预测框和所有真实框的**重合程度**，如果重合程度大于一定门限，则将**该预测框对应的先验框忽略**。其余作为负样本。

最终损失由三个部分组成：
1. 正样本，编码后的**长宽与xy轴偏移量与预测值的差距**。
2. 正样本，预测结果中**置信度的值与1对比**；负样本，预测结果中**置信度的值与0对比**。
3. 实际存在的框，**种类预测结果与实际结果的对比**。

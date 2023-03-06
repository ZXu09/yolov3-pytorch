
1. [预测部分](#一预测部分)
2. [主干特征提取网络darknet53介绍](#1主干特征提取网络darknet53介绍)
    - [残差网络](#残差网络)
    - [darknet53](#darknet53)
3. [从特征获取预测结果](#2从特征获取预测结果)
    - [构建FPN特征金字塔进行加强特征提取](#构建fpn特征金字塔进行加强特征提取)
    - [利用Yolo Head获得预测结果](#利用yolo-head获得预测结果)
4. [预测结果的解码](#3预测结果的解码)
    - [在原图上进行绘制](#4在原图上进行绘制)
5. [训练部分](#二训练部分)

生成目录：快捷键CTRL(CMD)+SHIFT+P，输入Markdown All in One: Create Table of Contents回车
# yolov3-pytorch
# 一、预测部分
## 1、主干特征提取网络darknet53介绍
输入416×416×3->进行下采样，宽高不断压缩，同时通道数不断扩张；若是进行上采样，宽高不断扩张，同时通道数不断压缩。

![darknet53](https://github.com/SZUZOUXu/yolov3-pytorch/blob/main/image/darknet53.jpg)

## 残差网络
### 梯度消失问题
**反向传播**更新参数时用的是**链式法则**，在很深的网络层中，由于**参数初始化一般更靠近0**，这样在训练的过程中更新浅层网络的参数时，很容易使得**梯度就接近于0**，而导致梯度消失，浅层的参数无法更新。

### 网络退化问题
- 随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当你再增加网络深度的话，**训练集loss反而会增大**。**梯度消失**则是导致网络退化的一个重要因素。
- 它**不是由过拟合产生的**，而是由**冗余的网络层**学习了**不是恒等映射的参数**造成的。

### 如何解决问题：
- 只把浅层的输出做**恒等映射**（即F(X)=0）输入到深层，这样网络加深也并不会出现网络退化。残差网络在更新梯度时把一些**乘法转变为了加法**。有效防止了梯度消失的情况
- **保证信息不丢失**。前向传输的过程中，随着层数的加深，Feature Map包含的图像信息会逐层减少，而ResNet的**直接映射**的加入，保证了 l+1 层的网络一定比 l 层包含**更多的图像信息**。

<div align=center>
<img src="https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/残差网络.png"/>
</div>

## darknet53
1. Darknet53具有一个重要特点是使用了**残差网络Residual**，
- Darknet53中的残差卷积首先进行一次卷积核大小为3X3、步长为2的卷积，该卷积会压缩**输入进来的特征层的宽和高**。此时我们可以获得一个特征层，我们将该**特征层命名为layer**。
- 再对该特征层进行一次**1X1的卷积下降通道数**，然后利用一个**3x3卷积提取特征并且上升通道数**。把这个结果**加上layer**，此时我们便构成了**残差结构**。

- 416,416,32 -> 208,208,64
- 3×3卷积，步长为2：416,416,32 -> 208,208,64
- 1×1卷积降维：208,208,64->208,208,32
- 3×3卷积升维：208,208,32->208,208,64
2. Darknet53的每一个卷积部分使用了特有的DarknetConv2D结构，每一次卷积的时候进行l2正则化，完成卷积后进行**BatchNormalization标准化与LeakyReLU**。
普通的ReLU是将所有的负值都设为零，**Leaky ReLU**则是给所有**负值赋予一个非零斜率**。

<div align=center>
<img src="https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/Darknetconv2D_BN_Leaky.png"/>
</div>

代码如下位置："yolo3-pytorch\nets\darknet.py"

## 2、从特征获取预测结果
三个特征层的shape分别为(52,52,256)、(26,26,512)、(13,13,1024)。

从特征获取预测结果的过程可以分为两个部分，分别是：

- 构建**FPN特征金字塔**进行加强**特征提取**。
- 利用**Yolo Head**对三个有效特征层进行**预测**。

### 构建FPN特征金字塔进行加强特征提取
- **上采样UmSampling2d**
上采样使用的方式为上池化，即元素复制扩充的方法使得特征图尺寸扩大
- **拼接concat**
concat 深层与浅层的特征图进行拼接
- **FPN(Feature Pyramid Networks)特征金字塔**
特征金字塔可以将不同shape的特征层进行**特征融合**，有利于**提取出更好的特征**。

1. 卷积网络中，随着网络**深度的增加**，特征图的**尺寸越来越小**，语义信息也越来越抽象，**感受野变大，检测大物体**。
2. 浅层特征图的语义信息较少，目标位置相对比较准确，深层特征图的语义信息比较丰富，目标位置则比较粗略，导致小物体容易检测不到。
3. FPN的功能可以说是**融合了浅层到深层的特征图** ，从而**充分利用各个层次的特征**。

### 利用Yolo Head(粉色框)获得预测结果
利用**FPN特征金字塔**，我们可以获得**三个加强特征**，这三个加强特征的shape分别为(13,13,512)、(26,26,256)、(52,52,128)，然后我们利用这三个shape的特征层传入Yolo Head获得预测结果。

- Yolo Head本质上是一次3x3卷积加上一次1x1卷积，**3x3卷积的作用是特征整合，1x1卷积的作用是调整通道数**。

1. (13,13,75)->(13,13,3,25)
13×13的网格，**每个网格3个先验框**（预先标注在图片上，预测结果判断先验框内部有无物体，对先验框进行中心还有宽高的调整）
2. (13,13,3,25)->(13,13,3,20+1+4)
VOC数据集分20个类，1判断先验框内是否有物体，4代表先验框的调整参数（4个参数确定一个框的位置）
3. 最后的维度应该为**75 = 3x25**

如果使用的是coco训练集，类则为80种，最后的维度应该为**255 = 3x85**

### 总结
其实际情况就是，输入N张416x416的图片，在经过多层的运算后，会输出三个shape分别为(N,13,13,75)，(N,26,26,75)，(N,52,52,75)的数据

对应每个图分为**13x13、26x26、52x52的网格上3个先验框的位置**。

代码如下位置："yolo3-pytorch\nets\yolo.py"

## 3、预测结果的解码
由第二步我们可以获得三个特征层的预测结果，shape分别为：

- (N,13,13,255)/(N,13,13,75)
- (N,26,26,255)/(N,13,13,75)
- (N,52,52,255)/(N,13,13,75)

在这里我们简单了解一下每个有效特征层到底做了什么：(注意**大的特征图**由于**感受野较小**，同时特征包含位置信息丰富，适合检测**小物体**。)

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

处理长宽不同的图片

对于很多分类、目标检测算法，输入的图片长宽是一样的，如224,224、416,416等。直接resize的话，图片就会失真。

但是我们可以采用如下的代码，使其用padding的方式不失真。
```Python
from PIL import Image
def letterbox_image(image, size):# size是想要的尺寸
    # 对图片进行resize，使图片不失真。在空缺的地方进行padding
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)# 取小边/原来的边（缩小系数），一般是缩小了尺寸（size<image.size）
    nw = int(iw*scale)//按缩小系数之后的边长，还保持原来的比例
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))# PIL.Image.new(mode, size, color)，用给定的模式和大小创建一个新图像，灰色填充
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))# //整数除法，向下取整（得到了应当复制过去的图像的左上角坐标）
    return new_image

img = Image.open("2007_000039.jpg")
new_image = letterbox_image(img,[416,416])
new_image.show()

```
# 二、训练部分
optimizer优化器：sgd
## 梯度下降法
### 批量梯度下降法
如果使用梯度下降法(批量梯度下降法)，那么每次迭代过程中都要对**n个样本**进行**求梯度**，所以开销非常大。
### 随机梯度下降法（stochastic gradient descent，SGD）
随机梯度下降的思想就是随机采样**一个样本**来更新参数

随机梯度下降虽然提高了计算效率，降低了计算开销，但是由于每次迭代只随机选择一个样本，因此随机性比较大，所以下降过程中非常曲折。
### 小批量梯度下降法
可以选取一定数目的样本组成一个**小批量样本**，然后用这个小批量更新梯度

lr_decay_type 学习率下降：cos

weight_decay：权值衰减，可防止过拟合

## 自适应矩估计（Adaptive Moment Estimation，Adam）
SGD 低效的根本原因是，梯度的方向并没有指向最小值的方向。为了改正SGD的缺点，引入了Adam。

梯度下降速度快，但是容易在最优值附近震荡。
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


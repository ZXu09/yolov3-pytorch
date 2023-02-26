# yolov3-pytorch
## 一、预测部分
主题网络darknet53介绍
![darknet53](https://github.com/SZUZOUXu/yolov3-pytorch/blob/main/image/darknet53.jpg)
1. Darknet53具有一个重要特点是使用了**残差网络Residual**，Darknet53中的残差卷积就是首先进行一次卷积核大小为3X3、步长为2的卷积，
该卷积会压缩输入进来的特征层的宽和高，此时我们可以获得一个特征层，我们将该特征层命名为layer。之后我们再对该特征层进行一次1X1的卷积和一次3X3的卷积，
并把这个结果加上layer，此时我们便构成了残差结构。

通过不断的1X1卷积和3X3卷积以及残差边的叠加，我们便大幅度的加深了网络。
**残差网络**的特点是容易优化，并且能够通过增加相当的深度来提高准确率。其内部的残差块使用了**跳跃连接**，缓解了在深度神经网络中增加深度带来的**梯度消失**问题。

2. Darknet53的每一个卷积部分使用了特有的DarknetConv2D结构，每一次卷积的时候进行l2正则化，完成卷积后进行BatchNormalization标准化与LeakyReLU。
普通的ReLU是将所有的负值都设为零，Leaky ReLU则是给所有负值赋予一个非零斜率。以数学的方式我们可以表示为：
![LeakyReLU](https://github.com/SZUZOUXu/yolov3-pytorch/blob/main/image/LeakyReLU.png)

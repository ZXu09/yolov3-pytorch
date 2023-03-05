# yolov3-pytorch

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 | 
|--|--|--|--|--|--|
| COCO-Train2017 | yolo_weights.pth | COCO-Val2017 | 416x416 | 38.0 | 67.2 |
## 所需环境
torch == 1.2.0 

详情请看requirements.txt
## 训练步骤
### a、训练VOC07+12数据集

1.  数据集的准备  
    **本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**
    
2.  数据集的处理  
    修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。
    
3.  开始网络训练  
    train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。
    
4.  训练结果预测  
    训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。  
    **model_path指向训练好的权值文件，在logs文件夹里。  
    classes_path指向检测类别所对应的txt。**  
    完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。
    
### b、训练自己的数据集

1.  数据集的准备  
    **本文使用VOC格式进行训练，训练前需要自己制作好数据集，**  
    训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
    训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
    
2.  数据集的处理  
    在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。  
    修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。  
    训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。  
    model_data/cls_classes.txt文件内容为：
    
```
cat
dog
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。

3.  开始网络训练  
    **训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
    **classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
    修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。
    
4.  训练结果预测  
    训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
    **model_path指向训练好的权值文件，在logs文件夹里。  
    classes_path指向检测类别所对应的txt。**  
    完成修改后就可以运行predict.py进行检测了。运行后输入图片路径（当前文件夹作为根目录）即可检测。

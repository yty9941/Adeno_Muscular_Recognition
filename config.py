import os

import torch


IMG_SIZE = 224  # 图像大小
IMG_MEAN = [0.250, 0.257, 0.263] # 图像均值和方差
IMG_STD = [0.220, 0.224, 0.227]

In_Channels = 3
Num_Classes = 1
# Threshold = 0.3604
Threshold = 0.5

Model_Backbone = "resnet50"  # ["resnet50", "ResT","EcaNet18","EcaNet34","CMT", "SwinTransformer"]
swinTransformerUrl = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
isCbam = False
isPretrained = True
Lamb = 0.01

# 超参数
Batch_Size = 32
Epochs = 100
Learning_Rate = 1e-5
Weight_Decay = 0.0005
Is_L1 = False # 是否使用L1正则化
Momentum = 0.9
Early_Stopping_Patience = 15 #早停法的容忍epoch

# 文件保存路径
Figure_Root = "/data/yangtongyu/Adeno_Muscular_Recognition/result/figures/"
Model_Root = "/data/yangtongyu/Adeno_Muscular_Recognition/result/models/"
if(isCbam):
    Figure_Save_Path = os.path.join(Figure_Root, Model_Backbone + "_CBAM", str(Epochs))
    Model_Save_Path = os.path.join(Model_Root, Model_Backbone + "_CBAM", str(Epochs))
else:
    Figure_Save_Path = os.path.join(Figure_Root, Model_Backbone, str(Epochs))
    Model_Save_Path = os.path.join(Model_Root, Model_Backbone, str(Epochs))
if(isPretrained):
    Figure_Save_Path += "_pretrained/"
    Model_Save_Path += "_pretrained/"

print(Figure_Save_Path, Model_Save_Path)
if not os.path.exists(Figure_Save_Path):
    os.makedirs(Figure_Save_Path)
if not os.path.exists(Model_Save_Path):
    os.makedirs(Model_Save_Path)

Gpu_Id = 0
if torch.cuda.is_available():  # GPU是否可用
    device = torch.device('cuda:' + str(Gpu_Id))
else:
    device = torch.device('cpu')
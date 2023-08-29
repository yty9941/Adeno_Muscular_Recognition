import numpy as np
import torch
import torchvision
from PIL import Image

from torch.backends import cudnn
import argparse
from torch.utils.data import Dataset
import pandas as pd
from utils import *

class MedicalDataset(Dataset): # 数据集
    def __init__(self, root_dir, item_list,  transform):
        super(MedicalDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.item_list = item_list


    def __getitem__(self, index):
        row = self.item_list.iloc[index]
        image1 = Image.open(os.path.join(self.root_dir, row.ImageRoot, row.Image1))
        image2 = Image.open(os.path.join(self.root_dir, row.ImageRoot, row.Image2))
        image3 = Image.open(os.path.join(self.root_dir, row.ImageRoot, row.Image3))
        label = row.Label

        image1 =  self.transform(image1)
        image2 =  self.transform(image2)
        image3 =  self.transform(image3)

        return image1, image2, image3
    def __len__(self):
        return len(self.item_list)

def get_mean_std(dataset):#计算三通道图像数据集的均值和方差
    means = [0, 0, 0]
    stds = [0, 0, 0]
    length = len(dataset) * 3
    print(length)
    image_list = []
    for image1, image2, image3 in dataset:
        image_list.append(image1)
        image_list.append(image2)
        image_list.append(image3)
    for image in image_list:
        for i in range(3):
            means[i] += image[i, :, :].mean()
            stds[i] += image[i, :, :].std()
    mean = np.array(means) / length
    std = np.array(stds) / length
    return mean, std
if __name__ == '__main__':
    root = "/data/yangtongyu/Adeno_Muscular_Recognition/data/"
    csv_path = root + "table.csv"
    data_list = pd.read_csv(csv_path)

    # 数据增强
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  ##将数据转换为张量，此处为img->tensor

    # 构建数据集train、val、test
    dataset = MedicalDataset(root, data_list, data_transform)
    print(dataset.__len__())
    mean, std = get_mean_std(dataset)
    print("mean", mean)
    print("std", std)
    # val_dataset = VideoDataset(args.root_dir, 'val', val_transform)
    # test_dataset = VideoDataset(args.root_dir, 'test', test_transform)


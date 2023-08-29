import os
from collections import defaultdict
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import config
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
        # image = torch.cat((image1, image2, image3), dim = 0)
        # print(os.path.join(self.root_dir, row.ImageRoot, row.Image1))
        # print("image1:",image1.size())
        # print("image2:",image2.size())
        # print("image3:",image3.size())
        # print("image:",image.size())
        # print("label", label)
        return image1, image2, image3, label
    def __len__(self):
        return len(self.item_list)

def dataloader(dataset):
    return DataLoader(dataset,
                    batch_size = config.Batch_Size,
                    shuffle = True,
                    pin_memory = True,
                    num_workers = 12)
if __name__ == '__main__':
    root = "/data/yangtongyu/Adeno_Muscular_Recognition/data/"
    csv_path = root + "train.csv"
    image_text_list = pd.read_csv(csv_path)
    transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = config.IMG_MEAN, std = config.IMG_STD)])
    dataset = MedicalDataset(root, image_text_list, transform)
    image1, label = dataset.__getitem__(0)



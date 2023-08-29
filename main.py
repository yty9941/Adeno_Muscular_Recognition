import torch
from torchvision import transforms
import pandas as pd
from dataset import MedicalDataset, dataloader
import config
from get_model import get_model
from model.ResNet import ResNet
from model.SwinTransformer import SwinTransformer
from torchsummary import summary
from utils import Utils
if __name__ == '__main__':
    trainer = Utils() # 声明工具类对象
    trainer.ensureReproduce()#确保每次训练结果均相同
    root = "/data/yangtongyu/Adeno_Muscular_Recognition/data/"
    train_csv_path = root + "train.csv"
    val_csv_path = root + "val.csv"
    train_list = pd.read_csv(train_csv_path)
    val_list = pd.read_csv(val_csv_path)
    train_transform = transforms.Compose([
        transforms.Resize((256, 334)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.ColorJitter(0.3, 0, 0, 0),
        # transforms.Resize((256, 256)),
        # transforms.CenterCrop((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.IMG_MEAN, std = config.IMG_STD)])
    val_transform = transforms.Compose([
        transforms.Resize((256, 334)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.IMG_MEAN, std = config.IMG_STD)])
    train_dataset = MedicalDataset(root, train_list, train_transform)
    val_dataset = MedicalDataset(root, val_list, val_transform)
    train_dataloader = dataloader(train_dataset)
    val_dataloader = dataloader(val_dataset)
    model = get_model(config.Model_Backbone)
    print(model)
    summary(model,(3,224,224), device = 'cpu')
    # 查看哪些参数计算梯度
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    if(config.isPretrained and config.Model_Backbone == "SwinTransformer"):
        swinTransformer = torch.utils.model_zoo.load_url(config.swinTransformerUrl)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in swinTransformer.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    trainer.train(model, train_dataloader, val_dataloader)

import torch
from torchvision import transforms
import pandas as pd
from dataset import MedicalDataset, dataloader
import config
from get_model import get_model
from model.ResNet import ResNet
from model.SwinTransformer import SwinTransformer
from utils import Utils
if __name__ == '__main__':
    tester = Utils() # 声明工具类对象
    tester.ensureReproduce()#确保每次训练结果均相同
    root = "/data/yangtongyu/Adeno_Muscular_Recognition/data/"
    test_csv_path = root + "test.csv"
    test_list = pd.read_csv(test_csv_path)
    test_transform = transforms.Compose([
        # transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.Resize((256, 334)),
        transforms.CenterCrop(config.IMG_SIZE),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.ColorJitter(0.2, 0.2),
        # transforms.RandomAffine(degrees = 10, scale = (0.8, 1.1), translate = (0.0625, 0.0625)),
        # transforms.Resize((256, 256)),
        # transforms.RandomCrop((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.IMG_MEAN, std = config.IMG_STD)])
    test_dataset = MedicalDataset(root, test_list, test_transform)
    test_dataloader = dataloader(test_dataset)
    model = get_model(config.Model_Backbone)
    # model = ResNet()
    # print(model)
    # 查看哪些参数计算梯度
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    model.load_state_dict(torch.load(config.Model_Save_Path + '/Best.pt'))
    tester.test(model, test_dataloader)

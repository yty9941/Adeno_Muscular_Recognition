import pandas as pd

import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import re
class dataProcessor:
    def __init__(self):
        super(dataProcessor, self).__init__()


    def splitDataset(self,item_list):
        item_list = item_list.to_numpy()

        seed = 8  # 设置随机数种子，使得image_path和Label打乱后仍对应
        np.random.seed(seed)
        np.random.shuffle(item_list)
        # print(image_text_list)
        # 70% for training , 20% for validation and 10% for testing
        train_list = item_list[0: int(len(item_list) * 0.8)]
        val_list = item_list[int(len(item_list) * 0.8): int(len(item_list) * 0.9)]
        test_list = item_list[int(len(item_list) * 0.9): len(item_list)]
        # 训练
        train_df = pd.DataFrame(data = train_list,
                                columns = ['ImageRoot', 'Image1', 'Image2', 'Image3', 'Class', 'Label'])
        train_df.to_csv(r'/data/yangtongyu/Adeno_Muscular_Recognition/data/train8.csv', index = False)
        # 验证
        val_df = pd.DataFrame(data = val_list,
                              columns = ['ImageRoot', 'Image1', 'Image2', 'Image3', 'Class', 'Label'])
        val_df.to_csv(r'/data/yangtongyu/Adeno_Muscular_Recognition/data/val8.csv', index = False)
        # 测试
        test_df = pd.DataFrame(data = test_list,
                               columns = ['ImageRoot', 'Image1', 'Image2', 'Image3', 'Class', 'Label'])
        test_df.to_csv(r'/data/yangtongyu/Adeno_Muscular_Recognition/data/test8.csv', index = False)


if __name__ == '__main__':
    csv_path = "/data/yangtongyu/Adeno_Muscular_Recognition/data/table.csv"
    item_list = pd.read_csv(csv_path)
    processor = dataProcessor()
    processor.splitDataset(item_list) # 直接划分数据集

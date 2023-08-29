import os
from collections import defaultdict
import pandas as pd

def generateCsv(root):
    classes = os.listdir(root)  # 分类目录 腺肌层和正常肌
    count_total = 0
    label = -1
    table = defaultdict(list)  # 对应表
    for _class in classes:
        label += 1
        annotators = os.listdir(os.path.join(root, _class))  # 标注师分类
        # print(annotators)
        for _annotator in annotators:
            patients = os.listdir(os.path.join(root, _class, _annotator))  # 病人分类
            for patient in patients:
                files = os.listdir(os.path.join(root, _class, _annotator, patient))
                image_list = []
                count = 0
                for file in files:
                    if (file.split(".")[-1] == "jpg" and count < 3):
                        image_list.append(file)
                        count += 1
                if (len(image_list) == 3):
                    count_total += 1
                    table["ImageRoot"].append(os.path.join(_class, _annotator, patient))
                    table["Image1"].append(image_list[0])
                    table["Image2"].append(image_list[1])
                    table["Image3"].append(image_list[2])
                    table["Class"].append(_class)
                    table["Label"].append(label)
                print(count_total, os.path.join(root, _class, _annotator, patient), image_list, _class, label)
    print(len(table["ImageRoot"]))
    dataset_df = pd.DataFrame(data = table,
                              columns = ['ImageRoot', 'Image1', 'Image2', 'Image3', 'Class', 'Label'])
    print(dataset_df)
    dataset_df.to_csv(r'/data/yangtongyu/Adeno_Muscular_Recognition/data/table.csv', index = False)

def readCsv(path):
    item_list = pd.read_csv(path)
    # print(item_list)
    label_list = item_list["Label"].to_list()
    count0 = 0
    count1 = 0
    total = len(label_list)

    for label in label_list:
        if(label == 0):
            count0 += 1
        if(label == 1):
            count1 += 1
    # print(total, count0, count1)
    print("总样本: "+ str(total) +
          "\n正常肌: "+ str(count0) +
          "\n腺肌症: "+ str(count1))







if __name__ == '__main__':
    root = "/data/yangtongyu/Adeno_Muscular_Recognition/data"
    generateCsv(root)
    # table_path = r"F:\数据集\data\table.csv"
    # readCsv(table_path)
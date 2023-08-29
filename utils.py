import random
from torch import nn
import os
import time
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from collections import defaultdict
import config
from matplotlib import pyplot as plt
from torch.backends import cudnn
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

from Early_Stopping import EarlyStopping


class Utils: # 工具类
    def __init__(self):
        # self.loss_fn = imageTextContrastiveLoss()
        self.loss_fn = nn.BCELoss()
        self.seed = 12

        pass
    def train(self, model, train_loader, val_loader):
        print("------Training Start!-------")
        train_start_time = time.time()
        model = model.to(config.device)
        # print(model)
        # print("--model:", model.device)
        optimizer = optim.Adam(model.parameters(), config.Learning_Rate,
                               betas = (0.9,0.98),eps = 1e-9,
                               weight_decay = config.Weight_Decay)



        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max = len(train_loader),
                                                       eta_min = 1e-8,
                                                       last_epoch = -1)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.98)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

        result_dict = defaultdict(list)
        min_loss = float('inf') #正无穷
        max_acc = 0
        early_stopping = EarlyStopping()
        total_epochs = 0 # 记录实际训练的epoch
        for epoch in range(1, config.Epochs + 1):
            epoch_start_time = time.time()
            model.train() # 训练

            batch_dict = defaultdict(list)
            # 打印当前学习率
            print("learning_rate: ",optimizer.state_dict()['param_groups'][0]['lr'])

            for batch in tqdm(train_loader):
                image1, image2, image3, label = batch
                image1 = image1.to(config.device)
                image2 = image2.to(config.device)
                image3 = image3.to(config.device)
                # image = image.to(config.device)
                label = label.to(config.device)
                output1 = model.forward(image1).squeeze(dim = -1)
                output2 = model.forward(image2).squeeze(dim = -1)
                output3 = model.forward(image3).squeeze(dim = -1)
                label_float = label.to(torch.float32).to(config.device)
                train_loss1 = self.loss_fn(output1, label_float)  # 计算损失
                train_loss2 = self.loss_fn(output2, label_float)  # 计算损失
                train_loss3 = self.loss_fn(output3, label_float)  # 计算损失
                train_loss = (train_loss1 + train_loss2 + train_loss3)/3

                predict_label1 = (output1 > config.Threshold).int().detach().cpu()
                predict_label2 = (output2 > config.Threshold).int().detach().cpu()
                predict_label3 = (output3 > config.Threshold).int().detach().cpu()

                predict_label = self.getPredictLabel(predict_label1, predict_label2, predict_label3)
                label_copy = label.clone().detach().cpu()

                train_acc_batch = accuracy_score(label_copy, predict_label)
                # 精确率
                train_precision_batch = precision_score(label_copy, predict_label)
                # 召回率
                train_recall_batch = recall_score(label_copy, predict_label)
                # f1-score
                train_f1_batch = f1_score(label_copy, predict_label)

                if (config.Is_L1):  # 是否使用L1正则化
                    L1_loss = 0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            L1_loss += torch.sum(torch.abs(param))
                    train_loss = 100 * train_loss + config.Lamb * L1_loss
                train_loss1 = train_loss.clone().detach().cpu()
                optimizer.zero_grad()  # 梯度清零
                torch.use_deterministic_algorithms(False)
                train_loss.backward()  # 后向传递过程
                optimizer.step()  # 更新（优化）权重与偏差矩阵
                batch_dict["train_loss"].append(train_loss1)
                batch_dict["train_acc"].append(train_acc_batch)
                batch_dict["train_precision"].append(train_precision_batch)
                batch_dict["train_recall"].append(train_recall_batch)
                batch_dict["train_f1"].append(train_f1_batch)

            scheduler.step()
            model.eval()  # 验证
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    image1, image2, image3, label = batch
                    image1 = image1.to(config.device)
                    image2 = image2.to(config.device)
                    image3 = image3.to(config.device)
                    # image = image.to(config.device)
                    label = label.to(config.device)
                    output1 = model.forward(image1).squeeze(dim = -1)
                    output2 = model.forward(image2).squeeze(dim = -1)
                    output3 = model.forward(image3).squeeze(dim = -1)
                    label_float = label.to(torch.float32).to(config.device)
                    val_loss1 = self.loss_fn(output1, label_float)  # 计算损失
                    val_loss2 = self.loss_fn(output2, label_float)  # 计算损失
                    val_loss3 = self.loss_fn(output3, label_float)  # 计算损失
                    val_loss = (val_loss1 + val_loss2 + val_loss3) / 3
                    label_copy = label.clone().detach().cpu()

                    predict_label1 = (output1 > config.Threshold).int().detach().cpu()
                    predict_label2 = (output2 > config.Threshold).int().detach().cpu()
                    predict_label3 = (output3 > config.Threshold).int().detach().cpu()
                    predict_label = self.getPredictLabel(predict_label1, predict_label2, predict_label3)

                    val_acc_batch = accuracy_score(label_copy, predict_label)
                    # 精确率
                    val_precision_batch = precision_score(label_copy, predict_label)
                    # 召回率
                    val_recall_batch = recall_score(label_copy, predict_label)
                    # f1-score
                    val_f1_batch = f1_score(label_copy, predict_label)
                    label_float = label.to(torch.float32).to(config.device)

                    if(config.Is_L1): # 是否使用L1正则化
                        L1_loss = 0
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                L1_loss += torch.sum(torch.abs(param))
                        # print("whether grad is True: ", L1_loss.requires_grad)
                        val_loss = 100 * val_loss + config.Lamb * L1_loss
                    val_loss1 = val_loss.clone().detach().cpu()
                    batch_dict["val_loss"].append(val_loss1)
                    batch_dict["val_acc"].append(val_acc_batch)
                    batch_dict["val_precision"].append(val_precision_batch)
                    batch_dict["val_recall"].append(val_recall_batch)
                    batch_dict["val_f1"].append(val_f1_batch)

                    # softmax = torch.nn.Softmax(dim = 1)
                    # probabilities = softmax(output)
                    # index = probabilities.argmax(dim = 1).squeeze().to(config.device)  # 概率最大的作为最终的标签
                    # val_correct += ((label == index) == True).sum()
                    # val_total += label.size(0)
                    # val_correct += ((label == predict_label) == True).sum()
                    # val_total += label.size(0)


            # acc_val = (val_correct * 1.0 / val_total).cpu().numpy()
            # train指标
            loss_train_mean = np.mean(np.array(batch_dict["train_loss"]))
            acc_train_mean = np.mean(np.array(batch_dict["train_acc"]))
            precision_train_mean = np.mean(np.array(batch_dict["train_precision"]))
            recall_train_mean = np.mean(np.array(batch_dict["train_recall"]))
            f1_train_mean = np.mean(np.array(batch_dict["train_f1"]))
            # val指标
            loss_val_mean = np.mean(np.array(batch_dict["val_loss"]))
            acc_val_mean = np.mean(np.array(batch_dict["val_acc"]))
            precision_val_mean = np.mean(np.array(batch_dict["val_precision"]))
            recall_val_mean = np.mean(np.array(batch_dict["val_recall"]))
            f1_val_mean = np.mean(np.array(batch_dict["val_f1"]))

            print("Epoch: {}/{}".format(epoch, config.Epochs))
            print(f"[ Train | {epoch:03d}/{config.Epochs:03d} ] \n"
                  f"train_loss = {loss_train_mean:.5f}, val_loss = {loss_val_mean:.5f} \n"
                  f"train_acc = {acc_train_mean:.5f}, val_acc = {acc_val_mean:.5f} \n"
                  f"train_precision = {precision_train_mean:.5f}, val_precision = {precision_val_mean:.5f} \n"
                  f"train_recall = {recall_train_mean:.5f}, val_recall = {recall_val_mean:.5f} \n"
                  f"train_f1 = {f1_train_mean:.5f}, val_f1 = {f1_val_mean:.5f} \n")
            if (loss_val_mean < min_loss):
                min_loss = loss_val_mean
                torch.save(model.state_dict(), config.Model_Save_Path + '/Best.pt')
                print("Epoch " + str(epoch) + " save model!")
            if (acc_val_mean > max_acc):
                max_acc = acc_val_mean
            epoch_end_time = time.time()
            epoch_time_interval = epoch_end_time - epoch_start_time
            print('耗时 {:.0f}m {:.0f}s'.format(epoch_time_interval // 60, epoch_time_interval % 60))

            # 记录
            result_dict["train_loss"].append(loss_train_mean)
            result_dict["train_acc"].append(acc_train_mean)
            result_dict["train_precision"].append(precision_train_mean)
            result_dict["train_recall"].append(recall_train_mean)
            result_dict["train_f1"].append(f1_train_mean)

            result_dict["val_loss"].append(loss_val_mean)
            result_dict["val_acc"].append(acc_val_mean)
            result_dict["val_precision"].append(precision_val_mean)
            result_dict["val_recall"].append(recall_val_mean)
            result_dict["val_f1"].append(f1_val_mean)
            early_stopping(loss_val_mean)
            if early_stopping.early_stop:
                print("Epoch " + str(epoch) + " early stopping！")
                total_epochs = epoch
                break
        print("------Training Finish!-------")
        total_epochs = total_epochs if total_epochs < config.Epochs else config.Epochs
        train_end_time = time.time()
        train_time_interval = train_end_time - train_start_time
        print('训练总耗时 {:.0f}m {:.0f}s'.format(train_time_interval // 60, train_time_interval % 60))
        print(f"min_loss = {min_loss:.5f}, max_acc = {max_acc:.5f}")
        self.drawFig(result_dict, "loss", total_epochs)
        self.drawFig(result_dict, "acc", total_epochs)
        self.drawFig(result_dict, "precision", total_epochs)
        self.drawFig(result_dict, "recall", total_epochs)
        self.drawFig(result_dict, "f1", total_epochs)


    def test(self, model, test_loader):
        print("------Test Start!-------")
        test_start_time = time.time()
        model = model.to(config.device)
        model.eval()  # 验证
        result_dict = defaultdict(list)

        # 用于绘制灰度混淆矩阵
        label_gt = []
        label_pred = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                image1, image2, image3, label = batch
                image1 = image1.to(config.device)
                image2 = image2.to(config.device)
                image3 = image3.to(config.device)
                label = label.to(config.device)
                label_gt += label.cpu().numpy().tolist()

                output1 = model.forward(image1).squeeze(dim = -1)
                output2 = model.forward(image2).squeeze(dim = -1)
                output3 = model.forward(image3).squeeze(dim = -1)
                label_float = label.to(torch.float32).to(config.device)
                test_loss1 = self.loss_fn(output1, label_float)  # 计算损失
                test_loss2 = self.loss_fn(output2, label_float)  # 计算损失
                test_loss3 = self.loss_fn(output3, label_float)  # 计算损失
                test_loss = (test_loss1 + test_loss2 + test_loss3) / 3

                predict_label1 = (output1 > config.Threshold).int().detach().cpu()
                predict_label2 = (output2 > config.Threshold).int().detach().cpu()
                predict_label3 = (output3 > config.Threshold).int().detach().cpu()
                predict_label = self.getPredictLabel(predict_label1, predict_label2, predict_label3)
                label_pred += predict_label
                label_copy = label.clone().detach().cpu()
                test_acc_batch = accuracy_score(label_copy, predict_label)
                # 精确率
                test_precision_batch = precision_score(label_copy, predict_label)
                # 召回率
                test_recall_batch = recall_score(label_copy, predict_label)
                # f1-score
                test_f1_batch = f1_score(label_copy, predict_label)

                if (config.Is_L1):  # 是否使用L1正则化
                    L1_loss = 0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            L1_loss += torch.sum(torch.abs(param))
                    # print("whether grad is True: ", L1_loss.requires_grad)
                    test_loss = 100 * test_loss + config.Lamb * L1_loss
                test_loss1 = test_loss.clone().detach().cpu()
                result_dict["test_loss"].append(test_loss1)
                result_dict["test_acc"].append(test_acc_batch)
                result_dict["test_precision"].append(test_precision_batch)
                result_dict["test_recall"].append(test_recall_batch)
                result_dict["test_f1"].append(test_f1_batch)

                # softmax = torch.nn.Softmax(dim = 1)
                # probabilities = softmax(output)
                # index = probabilities.argmax(dim = 1).squeeze().to(config.device)  # 概率最大的作为最终的标签
                # test_correct += ((label == index) == True).sum()
                # test_total += label.size(0)

                # test_correct += ((label == predict_label) == True).sum()
                # test_total += label.size(0)
        # acc_test = (test_correct * 1.0 / test_total).cpu().numpy()
        print("label_gt", label_gt)
        print("label_pred", label_pred)
        print("------Test Finish!-------")
        loss_test_mean = np.mean(np.array(result_dict["test_loss"]))
        acc_test_mean = np.mean(np.array(result_dict["test_acc"]))
        precision_test_mean = np.mean(np.array(result_dict["test_precision"]))
        recall_test_mean = np.mean(np.array(result_dict["test_recall"]))
        f1_test_mean = np.mean(np.array(result_dict["test_f1"]))
        self.drawConfusionMatrix(label_gt, label_pred, ['normal_muscle', 'adenomyosis'])
        print(f"test_loss = {loss_test_mean:.5f}\n"
              f"test_acc = {acc_test_mean:.5f}\n"
              f"test_precision = {precision_test_mean:.5f}\n"
              f"test_recall = {recall_test_mean:.5f}\n"
              f"test_f1 = {f1_test_mean:.5f}")
        with open(config.Figure_Save_Path + "testResult.txt", "w") as f:
            f.write(f"test_loss = {loss_test_mean:.5f}\n"
              f"test_acc = {acc_test_mean:.5f}\n"
              f"test_precision = {precision_test_mean:.5f}\n"
              f"test_recall = {recall_test_mean:.5f}\n"
              f"test_f1 = {f1_test_mean:.5f}")
        test_end_time = time.time()
        test_time_interval = test_end_time - test_start_time
        print('测试总耗时 {:.0f}m {:.0f}s'.format(test_time_interval // 60, test_time_interval % 60))

    # 绘制训练过程曲线
    def drawFig(self, result, name, epochs = None):
        # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
        if epochs is None:
            x = range(1, config.Epochs + 1)
        else:
            x = range(1, epochs + 1)

        # save_figure_path = os.path.join(config.Figure_Save_Path, args.model_name)
        if name == "loss":
            y1 = result["train_loss"]
            y2 = result["val_loss"]
            plt.cla()
            plt.title('Loss', fontsize = 20)
            line1, = plt.plot(x, y1)
            line2, = plt.plot(x, y2)
            plt.legend(handles = [line1, line2], labels = ["train", "val"], loc = "upper right", fontsize = 8)
            # plt.plot(x, y1, x, y2)
            plt.xlabel('epoch', fontsize = 20)
            plt.ylabel('loss', fontsize = 20)
            plt.grid()
            plt.savefig(config.Figure_Save_Path + "loss.png")
            plt.show()
        elif name == "acc":
            y1 = result["train_acc"]
            y2 = result["val_acc"]
            plt.cla()
            plt.title('Accuracy', fontsize = 20)
            # plt.plot(x, y1, x, y2)
            line1, = plt.plot(x, y1)
            line2, = plt.plot(x, y2)
            plt.legend(handles = [line1, line2], labels = ["train", "val"], loc = "lower right", fontsize = 8)
            plt.xlabel('epoch', fontsize = 20)
            plt.ylabel('acc', fontsize = 20)
            plt.grid()
            plt.savefig(config.Figure_Save_Path + "acc.png")
            plt.show()
        elif name == "precision":
            y1 = result["train_precision"]
            y2 = result["val_precision"]
            plt.cla()
            plt.title('Precision', fontsize = 20)
            # plt.plot(x, y1, x, y2)
            line1, = plt.plot(x, y1)
            line2, = plt.plot(x, y2)
            plt.legend(handles = [line1, line2], labels = ["train", "val"], loc = "lower right", fontsize = 8)
            plt.xlabel('epoch', fontsize = 20)
            plt.ylabel('precision', fontsize = 20)
            plt.grid()
            plt.savefig(config.Figure_Save_Path + "precision.png")
            plt.show()
        elif name == "recall":
            y1 = result["train_recall"]
            y2 = result["val_recall"]
            plt.cla()
            plt.title('Recall', fontsize = 20)
            # plt.plot(x, y1, x, y2)
            line1, = plt.plot(x, y1)
            line2, = plt.plot(x, y2)
            plt.legend(handles = [line1, line2], labels = ["train", "val"], loc = "lower right", fontsize = 8)
            plt.xlabel('epoch', fontsize = 20)
            plt.ylabel('recall', fontsize = 20)
            plt.grid()
            plt.savefig(config.Figure_Save_Path + "recall.png")
            plt.show()
        elif name == "f1":
            y1 = result["train_f1"]
            y2 = result["val_f1"]
            plt.cla()
            plt.title('F1-Score', fontsize = 20)
            # plt.plot(x, y1, x, y2)
            line1, = plt.plot(x, y1)
            line2, = plt.plot(x, y2)
            plt.legend(handles = [line1, line2], labels = ["train", "val"], loc = "lower right", fontsize = 8)
            plt.xlabel('epoch', fontsize = 20)
            plt.ylabel('f1', fontsize = 20)
            plt.grid()
            plt.savefig(config.Figure_Save_Path + "f1.png")
            plt.show()

    # 查看模型参数
    def queryModelParam(self, model):
        # for name, param in model.named_parameters():
        #     print(name, ":", param.size())  # 查看每一层参数的形状
        #     print(model.state_dict()[name])  # 查看每个参数权重值

        # # 查看哪些参数计算梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                 print(name)

    def freezeModel(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
        return model

    # 确保每次训练结果都一样
    def ensureReproduce(self):
        random.seed(self.seed)  # 为python设置随机种子
        np.random.seed(self.seed)  # 为numpy设置随机种子
        torch.manual_seed(self.seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(self.seed)  # 为当前GPU设置随机种子
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # 为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)

    # 绘制灰度混淆矩阵
    def drawConfusionMatrix(self, label_true, label_pred, label_name, title = "Confusion Matrix", dpi = 100):
        """

        @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
        @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
        @param label_name: 标签名字，比如['cat','dog','flower',...]
        @param title: 图标题
        @param save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
        @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
        @return:

        example：
                draw_confusion_matrix(label_true=y_gt,
                              label_pred=y_pred,
                              label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                              title="Confusion Matrix on Fer2013",
                              pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                              dpi=300)

        """

        cm = confusion_matrix(y_true = label_true, y_pred = label_pred, normalize = 'true')

        plt.imshow(cm, cmap = 'Blues')
        plt.title(title)
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(label_name.__len__()), label_name)
        plt.xticks(range(label_name.__len__()), label_name, rotation = 45)

        plt.tight_layout()

        plt.colorbar()

        for i in range(label_name.__len__()):
            for j in range(label_name.__len__()):
                color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
                value = float(format('%.2f' % cm[j, i]))
                plt.text(i, j, value, verticalalignment = 'center', horizontalalignment = 'center', color = color)
        plt.savefig(config.Figure_Save_Path + "confusion_matrix.png", bbox_inches = 'tight', dpi = dpi)
        plt.show()

    def getPredictLabel(self, predict_label1, predict_label2, predict_label3):
        predict_label1 = predict_label1.numpy().tolist()
        predict_label2 = predict_label2.numpy().tolist()
        predict_label3 = predict_label3.numpy().tolist()
        # print(len(predict_label1))
        # print(len(predict_label2))
        # print(len(predict_label3))
        predict_label = []
        for i in range(len(predict_label1)):
            count = 0
            if (predict_label1[i] == 0):
                count += 1
            if (predict_label2[i] == 0):
                count += 1
            if (predict_label3[i] == 0):
                count += 1
            if(count >= 2):
                predict_label.append(0)
            else:
                predict_label.append(1)
            # print(predict_label1[i])
        # print(predict_label)
        # a = input()
        return predict_label






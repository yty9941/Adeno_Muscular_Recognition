def train(self, model, train_loader, val_loader):
    print("------Training Start!-------")
    train_start_time = time.time()
    model = model.to(config.device)
    # print(model)
    # print("--model:", model.device)
    optimizer = optim.Adam(model.parameters(), config.Learning_Rate,
                           betas = (0.9, 0.98), eps = 1e-9,
                           weight_decay = config.Weight_Decay)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                T_max = len(train_loader),
    #                                                eta_min = 1e-8,
    #                                                last_epoch = -1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.98)

    result_dict = defaultdict(list)
    min_loss = float('inf')  # 正无穷
    max_acc = 0
    early_stopping = EarlyStopping()
    total_epochs = 0  # 记录实际训练的epoch
    for epoch in range(1, config.Epochs + 1):
        epoch_start_time = time.time()
        model.train()  # 训练
        train_total = 0
        train_correct = 0
        batch_dict = defaultdict(list)
        for batch in tqdm(train_loader):
            image, label = batch

            image = image.to(config.device)
            label = label.to(config.device)

            output = model.forward(image).squeeze(dim = -1)
            label_float = label.to(torch.float32).to(config.device)
            train_loss1 = self.loss_fn(output, label_float)  # 计算损失
            train_loss1 = self.loss_fn(output, label_float)  # 计算损失
            train_loss1 = self.loss_fn(output, label_float)  # 计算损失
            # print("output", output, output.size())
            # print("label_float", label_float, label_float.size() )
            # print("predict_label", predict_label, predict_label.size() )
            # print("train_loss", train_loss)
            predict_label = (output > config.Threshold).int().detach().cpu()
            label_copy = label.clone().detach().cpu()
            train_acc_batch = accuracy_score(label_copy, predict_label)
            # 精确率
            train_precision_batch = precision_score(label_copy, predict_label)
            # 召回率
            train_recall_batch = recall_score(label_copy, predict_label)
            # f1-score
            train_f1_batch = f1_score(label_copy, predict_label)
            # print("acc", train_acc_batch)
            # print("precision", train_precision_batch)
            # print("recall", train_recall_batch)
            # print("f1", train_f1_batch)
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

            # softmax = torch.nn.Softmax(dim = 1)
            # probabilities = softmax(output)
            # index = probabilities.argmax(dim = 1).squeeze().to(config.device)  # 概率最大的作为最终的标签
            #
            # train_correct += ((label == index) == True).sum()
            # train_total += label.size(0)
            # train_correct += ((label == predict_label) == True).sum()
            # train_total += label.size(0)

        # acc_train = (train_correct * 1.0 / train_total).cpu().numpy()
        # warmup for the first 10 epochs
        # if epoch >= 10:
        #     scheduler.step()
        scheduler.step()
        model.eval()  # 验证
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                image, label = batch
                image = image.to(config.device)
                label = label.to(config.device)

                # label1 = label.unsqueeze(1).unsqueeze(2)
                # image = image.unsqueeze(1).unsqueeze(2)
                # output = model.forward(image)
                # output_float = output.to(torch.float32).to(config.device)
                output = model.forward(image).squeeze(dim = -1)
                predict_label = (output > config.Threshold).int().detach().cpu()
                label_copy = label.clone().detach().cpu()
                val_acc_batch = accuracy_score(label_copy, predict_label)
                # 精确率
                val_precision_batch = precision_score(label_copy, predict_label)
                # 召回率
                val_recall_batch = recall_score(label_copy, predict_label)
                # f1-score
                val_f1_batch = f1_score(label_copy, predict_label)
                label_float = label.to(torch.float32).to(config.device)
                val_loss = self.loss_fn(output, label_float)  # 计算损失
                if (config.Is_L1):  # 是否使用L1正则化
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
        test_correct = 0
        test_total = 0
        # 用于绘制灰度混淆矩阵
        label_gt = []
        label_pred = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                image, label = batch
                image = image.to(config.device)
                label = label.to(config.device)
                # print("hhhhhhh", label.cpu().numpy(), label.size())
                label_gt += label.cpu().numpy().tolist()
                # label1 = label.unsqueeze(1).unsqueeze(2)
                # image = image.unsqueeze(1).unsqueeze(2)
                output = model.forward(image).squeeze(dim = -1)
                predict_label = (output > config.Threshold).int().detach().cpu()
                label_pred += predict_label.numpy().tolist()
                label_copy = label.clone().detach().cpu()
                test_acc_batch = accuracy_score(label_copy, predict_label)
                # 精确率
                test_precision_batch = precision_score(label_copy, predict_label)
                # 召回率
                test_recall_batch = recall_score(label_copy, predict_label)
                # f1-score
                test_f1_batch = f1_score(label_copy, predict_label)
                # predict_label = (output > config.Threshold).int()
                label_float = label.to(torch.float32).to(config.device)
                test_loss = self.loss_fn(output, label_float)  # 计算损失
                # output = model.forward(image)
                # output_float = output.to(torch.float32).to(config.device)
                # label_long = label.to(torch.long).to(config.device)
                # test_loss = self.loss_fn(output_float, label_long)  # 计算损失
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
        test_end_time = time.time()
        test_time_interval = test_end_time - test_start_time
        print('测试总耗时 {:.0f}m {:.0f}s'.format(test_time_interval // 60, test_time_interval % 60))
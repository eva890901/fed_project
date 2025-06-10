import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics

class DatasetSplit(Dataset):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = list(index)
    def __len__(self):
        return len(self.index)
    def __getitem__(self, item):
        image, label = self.dataset[self.index[item]]
        return image, label

class Local_Update(object):
    # 初始化客戶端的本地訓練環境 --> 設置loss function, DataLoader等等
    def __init__(self, args, dataset=None, index=None):
        # 儲存訓練參數 --> argparse+yaml(main.py)
        self.args = args
        # 設定loss fuction, 這裡是用Cross Entropy當loss function
        self.loss_function = nn.CrossEntropyLoss()
        # 儲存每輪被挑選中客戶端的index
        self.selected_clients = []
        # DataLoader把DataSplit所分割的客戶端資料子集依照args.loacl_bs的大小分批載入, 並啟用隨機打亂的選項提高訓練效率
        self.ldr_train = DataLoader(DatasetSplit(dataset, index), batch_size=self.args.local_bs, shuffle=True)
    
    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_index, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # 清空模型的梯度, 準備計算新的梯度
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_function(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_index % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_index * len(images), len(self.ldr_train.dataset),
                               100. * batch_index / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
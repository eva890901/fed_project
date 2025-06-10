import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import argparse
import yaml
import copy

from utils.sampling import Dirichlet_Distribution
from models.Update import Local_Update
from models.Nets import CNNmnist, CNNcifar
from models.Aggr_Algorithm import *
from models.test import test_img 

if __name__ == '__main__':
    
    # command-line argument管理 --> argparse+yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        for key in config:
            if (hasattr(args, key) == False):
                setattr(args, key, config[key])
    
    # 訓練模型的裝置
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # 資料集處理 --> MNIST(IID & Non-IID)與CIFAR10(IID)
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
        train_dataset = datasets.MNIST('../data/mnist', train=True, download=True, transform=trans_mnist)
        test_dataset = datasets.MNIST('../data/mnist', train=False, download=True, transform=trans_mnist)
        dataset_classes = np.unique(train_dataset.targets).tolist()
        if args.data_distribution == 'iid':
            alpha = float('inf')
            list_users = Dirichlet_Distribution(dataset_classes, train_dataset, alpha, args.num_users)
        elif args.data_distribution == 'noniid':
            alpha = args.alpha
            list_users = Dirichlet_Distribution(dataset_classes, train_dataset, alpha, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dataset_classes = np.unique(train_dataset.targets).tolist()
        if args.data_distribution == 'iid':
            alpha = float('inf')
            list_users = Dirichlet_Distribution(dataset_classes, train_dataset, alpha, args.num_users)
        else:
            exit('ERROR: Only consider IID situation in CIFAR10')
    else:
        exit('ERROR: unknown dataset')
    
    # 根據資料集初始化CNN模型
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNmnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNcifar(args=args).to(args.device)
    print(net_glob)
    net_glob.train()

    # 訓練模型時的變數初始化
    w_glob = net_glob.state_dict()
    training_loss = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    best_net = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        loss_locals = []
        # 利用dict來存挑選到庫戶端的資料, key: 挑中的客戶端index, value: 對應客戶端的更新參數
        w_locals = {}
        # 要求訓練人數必須至少有一人 --> 實驗預設num_users=200, fracs=0.1, 即每輪參與聚合的人數會是20人
        m = max(int(args.frac*args.num_users), 1) 
        # 挑選聯邦學習參與聚合動作的客戶端
        user_index = np.random.choice(range(args.num_users), m, replace=False)
        for index in user_index:
            local = Local_Update(args=args, dataset=train_dataset, index=list_users[index])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[index] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        w_glob = Combine_Metrics(w_locals)

        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        acc_train, loss_train_round = test_img(net_glob, train_dataset, args)
        net_glob.train()

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Train Acc {:.3f}'.format(iter, loss_avg, acc_train))
        training_loss.append(loss_avg)

    plt.figure()
    plt.plot(range(len(training_loss)), training_loss)
    plt.ylabel('Training Loss')
    plt.savefig('./save/fed_{}_{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac*args.num_users, args.data_distribution))

    net_glob.eval()
    training_accuarcy, training_loss = test_img(net_glob, train_dataset, args)
    testing_accuracy, testing_loss = test_img(net_glob, test_dataset, args)
    print("Training accuracy: {:.2f}".format(training_accuarcy))
    print("Testing accuracy: {:.2f}".format(testing_accuracy))
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import pickle
import pathlib
from tqdm import tqdm
import copy
import argparse

from numpy.random import RandomState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from preact_resnet import PreActResNet18

import numpy_indexed as npi

from data import *
from losses import *
from adversarial_attacks import *

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)


torch.autograd.set_detect_anomaly(True)

eps = 1e-8


parser = argparse.ArgumentParser(description='PyTorch MNIST Batch_Rewgt Training')
parser.add_argument('-dat', '--dataset', default="mnist",type=str, help="dataset")
parser.add_argument('-nr', '--noise_rate', default=0.4, type=float, help="noise rate")
parser.add_argument('-nt', '--noise_type', default="sym", type=str, help="noise type")
parser.add_argument('-loss', '--loss_name', default="cce",type=str, help="loss name")
parser.add_argument('-da', '--data_aug', default=0, type=int, help="data augmentation (0 or 1)")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-ep', '--num_epoch', default=100, type=int, help="number of epochs")
parser.add_argument('-run', '--num_runs', default=1, type=int, help="number of runs/simulations")
args = parser.parse_args()


def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc


def batch_rewgt_train(dataset_train, train_loader, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    idx_sel_tr_agg = np.zeros((len(train_loader.dataset), ))

    model.train()

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)

        if dataset == "news":
            x = x.type(torch.LongTensor)

        # if dataset == "svhn":
        #     x = x.reshape(-1, 3*32*32)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)

        # x = x.reshape((-1, 784))

        output = model(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)

        # batch_loss = nn.CrossEntropyLoss(reduction='mean')
        # loss_batch = batch_loss(output, y)
        loss_batch, idx_sel = loss_fn(output, y)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        optimizer.step()

        # loss_train += (torch.mean(batch_loss(output, y.to(device)))).item() # .item() for scalars, .tolist() in general

        # loss_train += torch.mean(loss_fn(output, y.to(device))).item()
        loss_train += torch.mean(loss_batch).item()
        correct += (pred.eq(y.to(device))).sum().item()

        idx_tmp = torch.zeros(x.shape[0])
        idx_tmp[idx_sel] = 1.
        idx_sel_tr_agg[list(map(int, idx.tolist()))] = np.asarray(idx_tmp.tolist())

        batch_cnt = batch_id + 1

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return loss_train, acc_train, idx_sel_tr_agg

def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt"
                            % (mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)
            if dataset == "news":
                x = x.type(torch.LongTensor)

            # if dataset == "svhn":
            #     x = x.reshape(-1, 3*32*32)
        
            x, y = x.to(device), y.to(device)

            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_batch, _ = loss_fn(output, y)
            loss_test += torch.mean(loss_batch).item()
            correct += (pred.eq(y.to(device))).sum().item()

            batch_cnt = batch_id + 1
        
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)

    return loss_test, acc_test


# Model
class MNIST_NN(nn.Module):
    def __init__(self, temp=1.0):
        super(MNIST_NN, self).__init__()

        # 1 I/P channel, 6 O/P channels, 5x5 conv. kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(400, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

        self.temp = temp
    
    def forward(self, x):

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) / self.temp
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CIFAR_NN(nn.Module):
    def __init__(self, temp=1.0):
        super(CIFAR_NN, self).__init__()

        # 1 I/P channel, 6 O/P channels, 5x5 conv. kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.temp = temp
    
    def forward(self, x):

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) / self.temp
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_small(nn.Module):
    def __init__(self, num_classes=10, temp=1.):
        super(CNN_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.temp = temp

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) / self.temp
        return x

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum 
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)        
        # self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)        
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)        
        # self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)        
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)        
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)        
        self.linear1=nn.Linear(256, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        # self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        # self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        # h=self.c2(h)
        # h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        # h=self.c4(h)
        # h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit


class CNN_CIFAR100(nn.Module):
    def __init__(self, input_channel=3, n_outputs=100, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum 
        super(CNN_CIFAR100, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)        
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)        
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)        
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)        
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)        
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)        
        self.linear1=nn.Linear(256, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit


class Net_dropout(nn.Module):

    """
    Adam - lr = 1e-3 used by https://www.kaggle.com/uvxy1234/cifar-10-implementation-with-pytorch
    """
    def __init__(self, dropout=0.2):
        super(Net_dropout, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        #3*32*32 -> 32*32*32
        self.dropout1 = nn.Dropout(p=dropout)        
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #32*32*32 -> 16*16*32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        #16*16*32 -> 16*16*64
        self.dropout2 = nn.Dropout(p=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #16*16*64 -> 8*8*64
        self.fc1 = nn.Linear(8*8*64, 1024)
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout4 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(512, 10)
            

    def forward(self, x):
        x = self.dropout1(self.conv1(x))
        x = self.pool1(F.relu(x))
        x = self.dropout2(self.conv2(x))
        x = self.pool2(F.relu(x))
        x = x.view(-1, self.num_flat_features(x)) 
        #self.num_flat_features(x) = 8*8*64 here.
        #-1 means: get the rest a row (in this case is 16 mini-batches)
        #pytorch nn only takes mini-batch as the input
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NewsNet(nn.Module):
    def __init__(self, weights_matrix, context_size=1000, hidden_size=300, num_classes=6):
        super(NewsNet, self).__init__()
        n_embed, d_embed = weights_matrix.shape

        print(f"\nn_embed, d_embed: {n_embed, d_embed}\n")
        self.embedding = nn.Embedding(n_embed, d_embed)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.Tensor(weights_matrix))

        self.avgpool=nn.AdaptiveAvgPool1d(16*hidden_size)
        self.fc1 = nn.Linear(16*hidden_size, 4*hidden_size)
        self.bn1=nn.BatchNorm1d(4*hidden_size)
        # self.avgpool=nn.AdaptiveAvgPool1d(4*hidden_size)
        # self.fc1 = nn.Linear(4*hidden_size, hidden_size)
        # self.bn1=nn.BatchNorm1d(hidden_size)
        # self.ac = nn.Softsign()
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.bn2=nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 84)
        self.bn3=nn.BatchNorm1d(84)
        self.fc4 = nn.Linear(84, num_classes)
        # self.fc2 = nn.Linear(hidden_size, 84)
        # self.bn2=nn.BatchNorm1d(84)
        # self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  
        embed = self.embedding(x) # input (128, 1000)
        embed = embed.detach()    # embed (128, 1000, 300)
        out = embed.view((1, embed.size()[0], -1)) # (1, 128, 300 000)
        out = self.avgpool(out)
        out = out.squeeze(0)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.ac(out)
        out = self.fc4(out)
        return out


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class SVHN_Net(nn.Module):
#     def __init__(self):
#         super(SVHN_Net, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
            
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             #nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             #nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0),
#             nn.MaxPool2d(2, 2), # output: 16 x 14 x 14
#             nn.Dropout(p=0.25),

#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
#             #nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
#             #nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 10, kernel_size=1, stride=1, padding=1),
#             #nn.MaxPool2d(2, 2), # output: 10 x 3 x 3

#             nn.AvgPool2d(kernel_size=6),           
#             nn.Flatten(),
#             nn.Softmax())
        
#     def forward(self, xb):
#         return self.network(xb)

class SVHN_Net(nn.Module):
    """
    Model adapted from 
    https://github.com/nanekja/JovianML-Project/blob/master/SVHN_Dataset_Classification_v2.ipynb
    """

    # def __init__(self, in_size=3*32*32, out_size=10):
    #     super().__init__()

    #     self.linear1 = nn.Linear(in_size, 64)
    #     self.linear2 = nn.Linear(64, 256)
    #     self.linear3 = nn.Linear(256, 32)
    #     self.linear4 = nn.Linear(32, out_size)

    #     self.bn1 = nn.BatchNorm1d(64)
    #     self.bn2 = nn.BatchNorm1d(256)
    #     self.bn3 = nn.BatchNorm1d(32)
        
    # def forward(self, xb):
    #     # Flatten the image tensors
    #     out = xb.view(xb.size(0), -1)
    #     out = self.linear1(out)
    #     out = self.bn1(F.relu(out)) # out = F.relu(out)
    #     out = self.linear2(out)
    #     out = self.bn2(F.relu(out)) # out = F.relu(out)
    #     out = self.linear3(out)
    #     out = self.bn3(F.relu(out)) # out = F.relu(out)
    #     out = F.relu(self.linear4(out))
    #     return out


    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # #nn.BatchNorm2d(256),
            # nn.ReLU(),
            
            # nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0),
            # nn.MaxPool2d(2, 2), # output: 16 x 14 x 14
            # nn.Dropout(p=0.25),

            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            # #nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            # #nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),

            # nn.Conv2d(256, 10, kernel_size=1, stride=1, padding=1),
            # #nn.MaxPool2d(2, 2), # output: 10 x 3 x 3
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 10, kernel_size=1, stride=3, padding=1),
            #nn.MaxPool2d(2, 2), # output: 10 x 3 x 3

            nn.AvgPool2d(kernel_size=6),           
            nn.Flatten(),
            nn.Softmax())
        
    def forward(self, xb):
        return self.network(xb)

    # # https://github.com/xingjunm/dimensionality-driven-learning/blob/master/models.py
    # def __init__(self):
    #     super(SVHN_Net, self).__init__()

    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=2)
    #     self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=2)

    #     self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
    #     self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.bn2 = nn.BatchNorm2d(64)

    #     self.fc1 = nn.Linear(300, 512)
    #     self.fc2 = nn.Linear(512, 128)
    #     self.fc3 = nn.Linear(128, 10)
    
    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = self.pool1(x)
    #     x = F.relu(self.bn2(self.conv1(x)))
    #     x = self.pool2(x)
    #     x = x.view(-1, 1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x



# If one wants to freeze layers of a network
# for param in model.parameters():
#   param.requires_grad = False


t_start = time.time()

"""
Configuration
"""

random_state = 422

dataset = args.dataset
noise_rate = args.noise_rate
noise_type = args.noise_type
data_aug = bool(args.data_aug)
batch_size = args.batch_size
loss_name = args.loss_name
num_epoch = args.num_epoch
num_runs = args.num_runs

# batch_size = 128

if dataset == "news":
    batch_size = 64

learning_rate = 2e-4

mode = "batch_rewgt" 

"""
Loss Function
"""

if loss_name == "gce":
    q = 0.7
    loss_fn = weighted_GCE(k=1, q=q, reduction="none")
elif loss_name == "cce":
    # loss_name = "cce"
    if dataset == "news":
        loss_fn = weighted_CCE(k=1, num_class=6, reduction="none")
    else:
        if dataset == "cifar100":
            loss_fn = weighted_CCE(k=1, num_class=100, reduction="none")
        elif dataset == "imagenet_tiny":
            loss_fn = weighted_CCE(k=1, num_class=200, reduction="none")
        else:
            loss_fn = weighted_CCE(k=1, reduction="none")
elif loss_name == "rll":
    if dataset == "news":
        loss_fn = weighted_RLL(k=1, alpha=0.45, num_class=6, reduction="none")
    else:
        loss_fn = weighted_RLL(k=1, alpha=0.45, reduction="none")

# loss_fn = nn.CrossEntropyLoss(reduction="none")

print("\n==============\nloss_name: {}\n=============\n".format(loss_name))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for run in range(num_runs):

    t_start = time.time()

    # if dataset == "cifar10" and run == 0 and noise_type == "sym":
    #     continue

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    best_acc_val = 0.

    chkpt_path = f"./checkpoint/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

    res_path = f"./results_pkl/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

    plt_path = f"./plots/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

    log_dirs_path = f"./runs/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if not os.path.exists(plt_path):
        os.makedirs(plt_path)

    if not os.path.exists(log_dirs_path):
        os.makedirs(log_dirs_path)
    else:
        for f in pathlib.Path(log_dirs_path).glob('events.out*'):
            try:
                f.unlink()
            except OSError as e:
                print(f"Error: {f} : {e.strerror}")
        print("\nLog files cleared...\n")

    print("\n============ PATHS =================\n")
    print(f"chkpt_path: {chkpt_path}")
    print(f"res_path: {res_path}")
    print(f"plt_path: {plt_path}")
    print(f"log_dirs_path: {log_dirs_path}")
    print("file name: %s-aug-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                mode, data_aug, dataset, 
                loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
    print("\n=============================\n")


    """
    Training/Validation/Test Data
    """

    dat, ids = read_data(noise_type, noise_rate, dataset, data_aug=data_aug, mode=mode)

    X_temp, y_temp, X_train, y_train = dat[0], dat[1], dat[2], dat[3]
    X_val, y_val, X_test, y_test = dat[4], dat[5], dat[6], dat[7]
    idx, idx_train, idx_val = ids[0], ids[1], ids[2]

    if noise_rate > 0.:
        idx_tr_clean_ref, idx_tr_noisy_ref = ids[3], ids[4]

    if dataset == "news":
        weights_matrix = dat[8]


    print("\n=============================\n")
    print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
    print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
    print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")    
    print("\n=============================\n")

    print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))


    """
    Create Dataset Loader
    """

    # Train. set
    tensor_x_train = torch.Tensor(X_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_y_train = torch.Tensor(y_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_id_train = torch.from_numpy(np.asarray(list(range(X_train.shape[0]))))

    dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train, tensor_id_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # Val. set
    tensor_x_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val)
    # tensor_id_val = torch.Tensor(idx_val)

    val_size = 1000
    dataset_val = torch.utils.data.TensorDataset(tensor_x_val, tensor_y_val) #, tensor_id_val)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=val_size, shuffle=True)

    # Test set
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    test_size = 1000
    dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_size, shuffle=True)


    """
    Initialize n/w and optimizer
    """

    if dataset == "mnist":
        model = MNIST_NN()
        # model = MLPNet()
    elif dataset in ["cifar10", "cifar100"]:

        if dataset == "cifar10":
            # model = CIFAR_NN()
            # model = CNN_small()
            model = CNN()
            # model = Net_dropout()
            # learning_rate = 1e-3
        else:
            model = CNN_CIFAR100()

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        alpha_plan = [learning_rate] * num_epoch
        beta1_plan = [mom1] * num_epoch
        for i in range(10, num_epoch):
            alpha_plan[i] = float(num_epoch - i) / (num_epoch - 10) * learning_rate
            beta1_plan[i] = mom2

        def adjust_learning_rate(optimizer, epoch):
            for param_group in optimizer.param_groups:
                param_group['lr']=alpha_plan[epoch]
                param_group['betas']=(beta1_plan[epoch], 0.999)
    
    elif dataset == "imagenet_tiny":
        model = PreActResNet18(num_classes=200)
        learning_rate = 1e-3

    elif dataset == "news":
        model = NewsNet(weights_matrix)
        learning_rate = 2e-4
    elif dataset == "svhn":
        model = SVHN_Net()
        learning_rate = 2e-4
    # params = list(model.parameters())
    model = model.to(device)
    print(model)

    """
    Optimizer and LR Scheduler

    Multiple LR Schedulers: https://github.com/pytorch/pytorch/pull/26423
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=0.1, patience=5, verbose=True, threshold=0.0001,
                    threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
    # lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)
    ## optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
    lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    # writer.add_graph(model, (tensor_x_train[0].unsqueeze(1)).to(device))
    # writer.close()

    epoch_idx_sel_tr_agg = np.zeros((len(train_loader.dataset), num_epoch))

    for epoch in range(num_epoch):

        #Training set performance
        loss_train, acc_train, idx_sel_tr_agg = batch_rewgt_train(dataset_train, train_loader, model)
        writer.add_scalar('training_loss', loss_train, epoch)
        writer.add_scalar('training_accuracy', acc_train, epoch)
        writer.close()
        # Validation set performance
        loss_val, acc_val = test(val_loader, model, run, use_best=False)
        #Testing set performance
        loss_test, acc_test = test(test_loader, model, run, use_best=False)
        writer.add_scalar('testing_loss', loss_test, epoch)
        writer.add_scalar('testing_accuracy', acc_test, epoch)
        writer.close()

        # Learning Rate Scheduler Update
        # if dataset == "mnist":
        #     # lr_scheduler_1.step(loss_val)
        #     ##lr_scheduler_3.step()
        #     lr_scheduler_2.step()
        # elif dataset in ["cifar10","cifar100"]:
        #     adjust_learning_rate(optimizer, epoch)

        if dataset in ["mnist", "svhn"]:
            lr_scheduler_1.step(loss_val)
        elif dataset in ["cifar10", "cifar100"]:
            lr_scheduler_1.step(loss_val)
            # lr_scheduler_2.step()
            # adjust_learning_rate(optimizer, epoch)
        elif dataset == "news":
            lr_scheduler_1.step(loss_val)
            # lr_scheduler_2.step()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)    

        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        epoch_idx_sel_tr_agg[:, epoch] = idx_sel_tr_agg

        # Update best_acc_val
        if epoch == 0:
            best_acc_val = acc_val


        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (
                                mode, dataset, noise_type, loss_name, str(int(noise_rate * 10)), str(run)))
            print("Model weights updated...\n")

        print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                    optimizer.param_groups[0]['lr'], 
                                                    loss_train, loss_val, loss_test, 
                                                    acc_train, acc_val, acc_test))


    # Test accuracy on the best_val MODEL
    loss_test, acc_test = test(test_loader, model, run, use_best=False)
    print("Test set performance - test_acc: {}, test_loss: {}\n".format(acc_test, loss_test))

    # Print the elapsed time
    elapsed = time.time() - t_start
    print("\nelapsed time: \n", elapsed)

    """
    Save results
    """


    with open(res_path+ "%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, loss_name, noise_type, 
                str(int(noise_rate * 10)), str(run)), 'wb') as f:
        if noise_rate > 0.: 
            pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                        'epoch_acc_train': np.asarray(epoch_acc_train), 
                        'epoch_loss_test': np.asarray(epoch_loss_test), 
                        'epoch_acc_test': np.asarray(epoch_acc_test), 
                        'y_train_org': y_temp[idx_train], 'y_train':y_train, 
                        'epoch_idx_sel_tr_agg': epoch_idx_sel_tr_agg, 
                        'idx_tr_clean_ref': idx_tr_clean_ref, 
                        'idx_tr_noisy_ref': idx_tr_noisy_ref,
                        'num_epoch': num_epoch,
                        'time_elapsed': elapsed}, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                'epoch_acc_train': np.asarray(epoch_acc_train), 
                'epoch_loss_test': np.asarray(epoch_loss_test), 
                'epoch_acc_test': np.asarray(epoch_acc_test), 
                'y_train_org': y_temp[idx_train], 'y_train':y_train, 
                'num_epoch': num_epoch,
                'time_elapsed': elapsed}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle file saved: " + res_path+ "%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, loss_name, 
                    noise_type, str(int(noise_rate * 10)), str(run)), "\n")

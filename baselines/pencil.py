from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import math
import pickle
from tqdm import tqdm
from collections import OrderedDict
import copy
import argparse

from numpy.random import RandomState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet, BasicBlock

from data import *
from adversarial_attacks import *
from losses import *

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
torch.cuda.manual_seed_all(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8

def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc

def pencil_train(epoch, train_loader, y_train_corr, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    K = 10.

    model.train()

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        x, y, idx = x.to(device), y.to(device), idx.to(device)
        y_one_hot = F.one_hot(y, num_classes=num_class)

        output = model(x)

        if epoch < phase_1 or epoch >= phase_2:
            loss_class = loss_fn(output, y)
            if epoch == phase_1 - 1:
                y_train_corr[list(map(int, idx.tolist())), :] = K * np.asarray(y_one_hot.tolist())

        elif epoch >= phase_1 and epoch < phase_2:
            y_tmp = torch.Tensor(y_train_corr[list(map(int, idx.tolist())), :]).to(device)
            y_tmp.requires_grad = True
            y_hat = F.softmax(y_tmp, dim=1)

            loss_class = (1./num_class) * torch.sum(F.softmax(output, dim=1) * (F.log_softmax(output, dim=1) 
                        - torch.log(y_hat)), dim=1)
            loss_compat = loss_fn(y_hat, y)
            loss_ent = -1. * torch.sum(F.softmax(output, dim=1) * F.log_softmax(output, dim=1), dim=1)
        
        if epoch < phase_1 or epoch >= phase_2:
            loss_batch = loss_class
        elif epoch >= phase_1 and epoch < phase_2:
            loss_batch = loss_class + (alpha * loss_compat) + (beta * loss_ent)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        optimizer.step()

        if epoch >= phase_1 and epoch < phase_2:
            # print(f"\nlab: {y_tmp[:5,:]}\n")
            # print(f"\nlab: {y_tmp.grad[:5,:]}\n")
            # input("\nWAIT....\n")
            with torch.no_grad():
                y_tmp.sub_(lamda*y_tmp.grad)
                y_train_corr[list(map(int, idx.tolist())), :] = np.asarray(y_tmp.tolist())

        
        # Compute the accuracy and loss for model_stud
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_train += loss_batch.mean().item()
        correct += pred.eq(y.to(device)).sum().item()

        batch_cnt = batch_id + 1

        loss_train_agg[list(map(int, idx.tolist()))] = np.asarray(loss_batch.tolist())
        acc_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.eq(y.to(device)).tolist())

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return loss_train, acc_train, loss_train_agg, acc_train_agg

def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load("%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, 
                                            loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)
        
            x, y = x.to(device), y.to(device)

            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_test += torch.mean(loss_fn(output, y)).item()
            correct += (pred.eq(y.to(device))).sum().item()

            batch_cnt = batch_id + 1
        
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)

    return loss_test, acc_test


# Model
class MNIST_ResNet18(ResNet):
    def __init__(self):
        super(MNIST_ResNet18, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)


class CIFAR_ResNet18(ResNet):
    def __init__(self):
        super(CIFAR_ResNet18, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)

# Model
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # 1 I/P channel, 6 O/P channels, 5x5 conv. kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(120)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(400, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, x):

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def lr_scheduler_pencil(epoch, optimizer):

    if epoch < phase_2:
        lr = learning_rate
    elif epoch < phase_2 + ((epoch - phase_2) //3):
        lr = learning_rate_2
    elif epoch < phase_2 + (2 * (epoch - phase_2) //3):
        lr = learning_rate_2 // 10
    else:
        lr = learning_rate_2 // 100

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

"""
Configuration
"""

num_epoch = 50
batch_size = 128
random_state = 422
num_runs = 1 # 5


dataset = "cifar10" # "checker_board" # "board" #  "cifar10"
data_aug = True
num_class = 10
mode = "pencil"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
PENCIL hyper-params
"""
phase_1 = 10
phase_2 = 35
learning_rate = 0.03 # for stage 1
learning_rate_2 = 1e-4
alpha = 0.01
beta = 0.1
# [200, 300, 400, 800, 1200] for sym. noise
# [600] for low cc noise
# [3000 -> 0] for high cc noise
lamda = 500



parser = argparse.ArgumentParser(description='PENCIL - CVPR 2019')
parser.add_argument('-nr','--noise_rate', default=0.4, type=float, help='noise_rate')
parser.add_argument('-nt','--noise_type', default="sym", type=str, help='noise_type')
parser.add_argument('-loss','--loss_name', default="cce", type=str, help='loss_name')
args = parser.parse_args()


noise_rate = args.noise_rate # 0.6
noise_type = args.noise_type # "sym"
loss_name = args.loss_name


for run in range(num_runs):

    print("\n==================\n")
    print(f"== RUN No.: {run} ==")
    print("\n==================\n")


    chkpt_path = "./checkpoint/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

    res_path = "./results_pkl/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

    plt_path = "./plots/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

    log_dirs_path = "./runs/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if not os.path.exists(plt_path):
        os.makedirs(plt_path)

    if not os.path.exists(log_dirs_path):
        os.makedirs(log_dirs_path)

    
    print("\n============ PATHS =================\n")
    print("chkpt_path: {}".format(chkpt_path))
    print("res_path: {}".format(res_path))
    print("plt_path: {}".format(plt_path))
    print("log_dirs_path: {}".format(log_dirs_path))
    print("\n=============================\n")

    """
    Read DATA
    """

    if dataset in ["board", "checker_board", "mnist", "cifar10", "cifar100"]:
        dat, ids = read_data(noise_type, noise_rate, dataset, data_aug=data_aug, mode=mode)

        X_temp, y_temp = dat[0], dat[1]
        X_train, y_train = dat[2], dat[3]
        X_val, y_val = dat[4], dat[5]
        X_test, y_test = dat[6], dat[7]

        if noise_rate > 0.:
            idx, idx_train, idx_val = ids[0], ids[1], ids[2]
            idx_tr_clean_ref, idx_tr_noisy_ref = ids[3], ids[4]
            idx_train_clean, idx_train_noisy = ids[5], ids[6]
        
        else:
            idx, idx_train, idx_val = ids[0], ids[1], ids[2]
            idx_train_clean, idx_train_noisy = ids[3], ids[4]

        if int(np.min(y_temp)) == 0:
            num_class = int(np.max(y_temp) + 1)
        else:
            num_class = int(np.max(y_temp))
    else:
        raise SystemExit("Dataset not supported.\n")


    print("\n=============================\n")
    print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
    print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
    print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")
    print("y_train - min : {}, y_val - min : {}, y_test - min : {}".format(np.min(y_train), np.min(y_val), np.min(y_test)))
    print("y_train - max : {}, y_val - max : {}, y_test - max : {}".format(np.max(y_train), np.max(y_val), np.max(y_test)))
    print("\n=============================\n")
    print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))

    """
    Create Dataset Loader
    """

    tensor_x_train = torch.Tensor(X_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_y_train = torch.Tensor(y_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_id_train = torch.from_numpy(np.asarray(list(range(X_train.shape[0]))))

    dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train, tensor_id_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    tensor_x_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val)

    val_size = y_val.shape[0]
    dataset_val = torch.utils.data.TensorDataset(tensor_x_val, tensor_y_val)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=val_size, shuffle=True)

    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    test_size = 1000
    dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_size, shuffle=True)



    """
    Choose MODEL and LOSS FUNCTION
    """
    
    if dataset == "mnist":
        # model = MNIST_CNN()
        model = MNIST_ResNet18()
    elif dataset in ["cifar10", "cifar100"]:
        # model = torchvision.models.resnet18()
        model = CIFAR_ResNet18()

    model = model.to(device)
    print(model)

    print("\n===========\nloss: {}\n===========\n".format(loss_name))

    if loss_name == "cce":
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    elif loss_name == "cce_new":
        loss_fn = CCE_new_APL(num_class=num_class, reduction="none")
    elif loss_name == "gce":
        q = 0.7
        loss_fn = GCE(q=q, num_class=num_class, reduction="none")
    elif loss_name == "dmi":
        loss_fn = L_DMI(num_class=num_class)
        
        model.load_state_dict(torch.load(chkpt_path + "%s-%s-cce-%s-nr-0%s-mdl-wts-run-%s.pt"
                                % (mode, dataset, noise_type, str(int(noise_rate * 10)), str(run))))
    elif loss_name == "rll":
        alpha = 0.45 # 0.45/0.5/0.6 - works well with lr = 3e-3
        loss_fn = RLL(alpha=alpha, num_class=num_class, reduction="none")
    elif loss_name == "mae":
        loss_fn = MAE(num_class=num_class, reduction="none")
    elif loss_name == "mse":
        loss_fn = MSE(num_class=num_class, reduction="none")
    elif loss_name == "norm_mse":
        loss_fn = MSE_APL(num_class=num_class, reduction="none")
    else:
        raise SystemExit("Invalid loss function\n")


    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    writer.add_graph(model, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
    writer.close()


    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    y_train_corr = np.zeros((len(train_loader.dataset), num_class))

    best_acc_val = 0.

    t_start = time.time()

    for epoch in range(num_epoch):

        #Training set performance
        loss_train, acc_train, loss_train_agg, acc_train_agg = pencil_train(epoch, train_loader, y_train_corr, model)
        writer.add_scalar('training_loss', loss_train, epoch)
        writer.add_scalar('training_accuracy', acc_train, epoch)
        writer.close()
        

        # Validation set performance
        loss_val, acc_val = test(val_loader, model, run, use_best=False)

        #Test set performance
        loss_test, acc_test = test(test_loader, model, run, use_best=False)
        writer.add_scalar('test_loss', loss_test, epoch)
        writer.add_scalar('test_accuracy', acc_test, epoch)
        writer.close()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)
        epoch_loss_train_agg[:, epoch] = loss_train_agg
        epoch_acc_train_agg[:, epoch] = acc_train_agg

        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        # Learning Rate Scheduler Update
        lr_scheduler_pencil(epoch, optimizer)

        # Update best_acc_val and sample_wts_fin
        if epoch == 0:
            best_acc_val = acc_val

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, 
                                loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
            print("Model weights updated...\n")

        print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                    optimizer.param_groups[0]['lr'], 
                                                    loss_train, loss_val, loss_test, 
                                                    acc_train, acc_val, acc_test))


    loss_test, acc_test = test(test_loader, model, run, use_best=False)

    print(f"Run: {run}::: Test set performance \
            - test_acc: {acc_test}, test_loss: \
            {loss_test}\n")

    if noise_rate > 0.:
        torch.save(model.state_dict(), chkpt_path + 
                    "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (
                    mode, dataset, loss_name, noise_type, 
                    str(int(noise_rate * 10)), str(run)))

    # model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt"
    #                     % (mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
    # model = model.to(device)

    """
    Adversarial Attacks
    """
    # if dataset=="mnist":
    #     epsilon = 0.1
    #     k = 40
    #     # alpha = 2.5 * epsilon / k
    
    # adv_acc_fgsm = test_fgsm_batch_ce(model, test_loader, epsilon)
    # adv_acc_pgd = test_pgd_batch_ce(model, test_loader, epsilon, k=k)

    # print("Checking performance for FGSM & PGD Adversarial attacks...\n")
    # print(f"Run {run} ::: FGSM Attack: acc. = {adv_acc_fgsm}, PGD Attack: acc. = {adv_acc_pgd}\n")

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
                            'epoch_loss_train_agg': epoch_loss_train_agg,
                            'epoch_acc_train_agg': epoch_acc_train_agg,
                            'y_train_corr': y_train_corr,
                            'idx_train': idx_train, 'idx_tr_clean_ref': idx_tr_clean_ref, 
                            'idx_tr_noisy_ref': idx_tr_noisy_ref,
                            'X_temp': X_temp,  'X_train': X_train, 
                            'X_val': X_val, 'y_temp': y_temp, 
                            'y_train':y_train, 'y_val':y_val,
                            'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                #             'epoch_acc_train': np.asarray(epoch_acc_train),
                #             'epoch_loss_test': np.asarray(epoch_loss_test), 
                #             'epoch_acc_test': np.asarray(epoch_acc_test), 
                #             'epoch_loss_train_agg': epoch_loss_train_agg,
                #             'epoch_acc_train_agg': epoch_acc_train_agg,
                #             'y_train_corr': y_train_corr,                
                #             'idx_train': idx_train, 'idx_tr_clean_ref': idx_tr_clean_ref, 
                #             'idx_tr_noisy_ref': idx_tr_noisy_ref,
                #             'X_temp': X_temp,  'X_train': X_train, 
                #             'X_val': X_val, 'y_temp': y_temp, 
                #             'y_train':y_train, 'y_val':y_val,
                #             'num_epoch': num_epoch, 'adv_acc_fgsm': adv_acc_fgsm, 
                #             'adv_acc_pgd': adv_acc_pgd}, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                            'epoch_acc_train': np.asarray(epoch_acc_train),
                            'epoch_loss_test': np.asarray(epoch_loss_test), 
                            'epoch_acc_test': np.asarray(epoch_acc_test), 
                            'epoch_loss_train_agg': epoch_loss_train_agg,
                            'epoch_acc_train_agg': epoch_acc_train_agg,
                            'X_temp': X_temp, 'X_train': X_train, 'X_val': X_val,
                            'y_temp': y_temp, 'y_train':y_train, 'y_val':y_val,
                            'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                #             'epoch_acc_train': np.asarray(epoch_acc_train),
                #             'epoch_loss_test': np.asarray(epoch_loss_test), 
                #             'epoch_acc_test': np.asarray(epoch_acc_test), 
                #             'epoch_loss_train_agg': epoch_loss_train_agg,
                #             'epoch_acc_train_agg': epoch_acc_train_agg,
                #             'X_temp': X_temp, 'X_train': X_train, 'X_val': X_val,
                #             'y_temp': y_temp, 'y_train':y_train, 'y_val':y_val,
                #             'num_epoch': num_epoch, 'adv_acc_fgsm': adv_acc_fgsm, 
                #             'adv_acc_pgd': adv_acc_pgd}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Pickle file saved: " + res_path+ "%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, loss_name, 
                    noise_type, str(int(noise_rate * 10)), str(run)), "\n")


    # Print the elapsed time
    elapsed = time.time() - t_start
    print("\nelapsed time: \n", elapsed)

    """
    Plot results
    """
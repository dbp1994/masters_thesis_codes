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
# tf.set_random_seed(3459)


torch.autograd.set_detect_anomaly(True)


eps = 1e-8

def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc

def rm_train(train_loader, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    model.train()

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):


        y = y.type(torch.LongTensor)
        x, y, idx = x.to(device), y.to(device), idx.to(device)

        output = model(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_batch = loss_fn(output, y)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        optimizer.step()

        loss_train += torch.mean(loss_batch).item()
        correct += (pred.eq(y.to(device))).sum().item()

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

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()

        # 1 I/P channel, 6 O/P channels, 5x5 conv. kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# If one wants to freeze layers of a network
# for param in model.parameters():
#   param.requires_grad = False


t_start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
Configuration
"""

num_epoch = 50
batch_size = 128
random_state = 422
num_runs = 1 # 5
# learning_rate = 2e-4
learning_rate = 0.25

dataset = "mnist" # "checker_board" # "board" # "cifar10" # "cifar100"
num_class = 10
mode = "rm"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Empirical Risk Minimization')
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
        dat, ids = read_data(noise_type, noise_rate, dataset, mode=mode)

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
        model = torchvision.models.resnet18()

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


    # optimizer = optim.Adam(model.params(), learning_rate)
    optimizer = optim.SGD(model.params(), lr = learning_rate, momentum=0.9, weight_decay=1e-4)

    lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.1, patience=5, verbose=True, threshold=0.0001,
                threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
    lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    # lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    writer.add_graph(model, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
    writer.close()

    best_acc_val = 0.

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))

    t_start = time.time()

    for epoch in range(num_epoch):

        #Training set performance
        loss_train, acc_train, loss_train_agg, acc_train_agg = rm_train(train_loader, model)
        writer.add_scalar('training_loss', loss_train, epoch)
        writer.add_scalar('training_accuracy', acc_train, epoch)
        writer.close()

        # Validation set performance
        loss_val, acc_val = test(val_loader, loss_fn, model, use_best=False)
        
        #Testing set performance
        loss_test, acc_test = test(test_loader, loss_fn, model, use_best=False)
        writer.add_scalar('testing_loss', loss_test, epoch)
        writer.add_scalar('testing_accuracy', acc_test, epoch)
        writer.close()

        # Learning Rate Scheduler Update
        # lr_scheduler_1.step(loss_val)
        ##lr_scheduler_3.step()
        ##lr_scheduler_2.step()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)
        epoch_loss_train_agg[:, epoch] = loss_train_agg
        epoch_acc_train_agg[:, epoch] = acc_train_agg    

        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        # Update best_acc_val
        if epoch == 1:
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


    # Test accuracy on the best_val MODEL
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
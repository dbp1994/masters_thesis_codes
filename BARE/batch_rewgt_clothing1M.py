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
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import numpy_indexed as npi

from data_clothing1M import *
from losses_clothing1M import *
from adversarial_attacks import *

# set seed for reproducibility
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8


parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Batch_Rewgt Training')
parser.add_argument('-loss', '--loss_name', default="cce",type=str, help="loss name")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-ep', '--num_epoch', default=15, type=int, help="number of epochs")
parser.add_argument('-run', '--num_runs', default=1, type=int, help="number of runs/simulations")
args = parser.parse_args()


def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc


def batch_rewgt_train(train_loader, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    model.train()

    for batch_id, (x, y) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y = x.to(device), y.to(device)

        # x = x.reshape((-1, 784))

        output = model(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)

        # batch_loss = nn.CrossEntropyLoss(reduction='mean')
        # loss_batch = batch_loss(output, y)
        loss_batch, _ = loss_fn(output, y)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        optimizer.step()

        # loss_train += (torch.mean(batch_loss(output, y.to(device)))).item() # .item() for scalars, .tolist() in general

        # loss_train += torch.mean(loss_fn(output, y.to(device))).item()
        loss_train += torch.mean(loss_batch).item()
        correct += (pred.eq(y.to(device))).sum().item()

        batch_cnt = batch_id + 1

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return loss_train, acc_train

def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-mdl-wts-run-%s.pt" % (mode, dataset, loss_name, str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)
        
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


t_start = time.time()

"""
Configuration
"""

random_state = 422
dataset = "clothing1M"
batch_size = args.batch_size
loss_name = args.loss_name
num_epoch = args.num_epoch
num_runs = args.num_runs
learning_rate = 1e-3
mode = "batch_rewgt_clothing1M" 
num_class = 14

"""
Loss Function
"""

if loss_name == "gce":
    q = 0.7
    loss_fn = weighted_GCE(q=q, k=1, num_class=num_class, reduction="none")
elif loss_name == "cce":
    # loss_name = "cce"
    loss_fn = weighted_CCE(k=1, num_class=num_class, reduction="none")
else:
    raise NotImplementedError(f"Batch Reweighting not implemented for - {loss_name}")

print("\n==============\nloss_name: {}\n=============\n".format(loss_name))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for run in range(num_runs):

    t_start = time.time()

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    best_acc_val = 0.

    chkpt_path = f"./checkpoint/{mode}/{dataset}/run_{str(run)}/"

    res_path = f"./results_pkl/{mode}/{dataset}/run_{str(run)}/"

    plt_path = f"./plots/{mode}/{dataset}/run_{str(run)}/"

    log_dirs_path = f"./runs/{mode}/{dataset}/run_{str(run)}/"

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
    print("file name: %s-%s-%s-run-%s.pt" % (mode, dataset, loss_name, str(run)))
    print("\n=============================\n")


    """
    Training/Validation/Test Data
    """

    # # dat, ids = read_data(noise_type, noise_rate, dataset, data_aug, mode)

    # # X_temp, y_temp, X_train, y_train = dat[0], dat[1], dat[2], dat[3]
    # # X_val, y_val, X_test, y_test = dat[4], dat[5], dat[6], dat[7]
    # # idx, idx_train, idx_val = ids[0], ids[1], ids[2]


    # print("\n=============================\n")
    # print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
    # print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
    # print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")    
    # print("\n=============================\n")

    """
    Create Dataset Loader
    """

    data_loader = clothing_dataloader(batch_size=batch_size, shuffle=True)
    train_loader, val_loader, test_loader = data_loader.run()

    print(train_loader.dataset, "\n", len(train_loader.dataset), "\n")

    input("\nPress <ENTER> to continue.\n")

    """
    Initialize n/w and optimizer
    """

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_class)
    model = model.to(device)
    print(model)

    """
    Optimizer
    """
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 11], gamma=0.5)


    """
    Use this optimizer and data config. for Clothing-1M, Animal-10N, Food-101N
    https://openreview.net/pdf?id=ZPa2SyGcbwh
    """

    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)

    for epoch in range(num_epoch):

        #Training set performance
        loss_train, acc_train = batch_rewgt_train(train_loader, model)
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

        # if epoch >= 40:
        #     learning_rate /= 10

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate

        lr_scheduler.step()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)    

        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        # Update best_acc_val
        if epoch == 0:
            best_acc_val = acc_val


        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-mdl-wts-run-%s.pt" % (mode, dataset, loss_name, str(run)))
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
    with open(res_path+ "%s-%s-%s-run-%s.pickle" % (mode, dataset, loss_name, str(run)), 'wb') as f:
        pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                    'epoch_acc_train': np.asarray(epoch_acc_train), 
                    'epoch_loss_test': np.asarray(epoch_loss_test), 
                    'epoch_acc_test': np.asarray(epoch_acc_test),
                    'num_epoch': num_epoch,
                    'time_elapsed': elapsed}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle file saved: " + res_path+ "%s-%s-%s-run-%s.pickle" % (mode, dataset, loss_name, str(run)), "\n")

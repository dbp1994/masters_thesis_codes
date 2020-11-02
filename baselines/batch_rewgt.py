from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import pickle
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
parser.add_argument('--nr', default=0.4, type=float, help='noise_rate')
parser.add_argument('--nt', default="sym", type=str, help='noise_type')
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

    model.train()

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)

        # x = x.reshape((-1, 784))

        output = model(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)

        # batch_loss = nn.CrossEntropyLoss(reduction='mean')
        # loss_batch = batch_loss(output, y)
        loss_batch = loss_fn(output, y)

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
                model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt"
                            % (mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)
        
            x, y = x.to(device), y.to(device)

            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_batch = loss_fn(output, y)
            loss_test += torch.mean(loss_batch).item()
            correct += (pred.eq(y.to(device))).sum().item()

            batch_cnt = batch_id + 1
        
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)

    return loss_test, acc_test


# Model
class MNIST_NN(nn.Module):
    def __init__(self):
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
    
    def forward(self, x):

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
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

class CIFAR10_NN(nn.Module):
    def __init__(self):
        super(CIFAR10_NN, self).__init__()

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

"""
Configuration
"""

num_epoch = 100
batch_size = 128
learning_rate = 2e-4

random_state = 422
dataset = "mnist" #"cifar10"

noise_rate = args.nr #0.6
noise_type = args.nt #"cc" #"sym"


mode = "batch_rewgt" 

"""
Loss Function
"""
if mode == "rm":
    if loss_name == "gce":
        q = 0.7
        loss_fn = GCE_loss(q=q, reduction="none")
    elif loss_name == "dmi":
        # loss_fn = L_DMI()
        model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-cce-nr-0%s-mdl-wts.pt"
                            % (mode, dataset, noise_type, str(int(noise_rate * 10)))))
        pass
    elif loss_name == "rll":
        pass
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
elif mode == "pencil":
    loss_name = "pencil"
    # loss_fn = pencil(alpha=alpha, beta=beta)
    pass
elif mode == "batch_rewgt":
    if loss_name == "gce":
        q = 0.7
        loss_fn = weighted_GCE(q=q, reduction="none")
    else:
        # loss_name = "cce"
        loss_fn = weighted_CCE(reduction="none")
else:
    loss_name = "cce"
    loss_fn = nn.CrossEntropyLoss(reduction="none")

print("\n==============\nloss_name: {}\n=============\n".format(loss_name))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


chkpt_path = "./checkpoint/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/"

res_path = "./results_pkl/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/"

plt_path = "./plots/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/"

log_dirs_path = "./runs/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/"

if not os.path.exists(chkpt_path):
    os.makedirs(chkpt_path)

if not os.path.exists(res_path):
    os.makedirs(res_path)

if not os.path.exists(plt_path):
    os.makedirs(plt_path)

if not os.path.exists(log_dirs_path):
    os.makedirs(log_dirs_path)


"""
Training/Validation/Test Data
"""

dat, ids = read_data(noise_type, noise_rate, dataset, mode)

X_temp, y_temp, X_train, y_train = dat[0], dat[1], dat[2], dat[3]
X_val, y_val, X_test, y_test = dat[4], dat[5], dat[6], dat[7]
idx, idx_train, idx_val = ids[0], ids[1], ids[2]


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
tensor_id_train = torch.Tensor(idx_train) # .as_tensor() avoids copying, .Tensor() creates a new copy

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
elif dataset == "cifar10":
    model = CIFAR10_NN()
# params = list(model.parameters())
model = model.to(device)
print(model)

"""
Setting up Tensorbard
"""
writer = SummaryWriter(log_dirs_path)
writer.add_graph(model, (tensor_x_train[0].unsqueeze(1)).to(device))
writer.close()


#Optimizer and LR Scheduler
"""
Multiple LR Schedulers: https://github.com/pytorch/pytorch/pull/26423
"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.1, patience=5, verbose=True, threshold=0.0001,
                threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
## optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


epoch_loss_train = []
epoch_acc_train = []
epoch_loss_test = []
epoch_acc_test = []

best_acc_val = 0.

for epoch in range(1, num_epoch+1):

    #Training set performance
    loss_train, acc_train = batch_rewgt_train(dataset_train, train_loader, model)
    writer.add_scalar('training_loss', loss_train, epoch)
    writer.add_scalar('training_accuracy', acc_train, epoch)
    writer.close()
    # Validation set performance
    loss_val, acc_val = test(val_loader, model, use_best=False)
    #Testing set performance
    loss_test, acc_test = test(test_loader, model, use_best=False)
    writer.add_scalar('testing_loss', loss_test, epoch)
    writer.add_scalar('testing_accuracy', acc_test, epoch)
    writer.close()

    # Learning Rate Scheduler Update
    lr_scheduler_1.step(loss_val)
    ##lr_scheduler_3.step()
    ##lr_scheduler_2.step()

    epoch_loss_train.append(loss_train)
    epoch_acc_train.append(acc_train)    

    epoch_loss_test.append(loss_test)
    epoch_acc_test.append(acc_test)

    # Update best_acc_val
    if epoch == 1:
        best_acc_val = acc_val


    if acc_val > best_acc_val:
        best_acc_val = acc_val
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts.pt" % (
                            mode, dataset, noise_type, loss_name, str(int(noise_rate * 10))))
        print("Model weights updated...\n")

    print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                optimizer.param_groups[0]['lr'], 
                                                loss_train, loss_val, loss_test, 
                                                acc_train, acc_val, acc_test))


# Test accuracy on the best_val MODEL
loss_test, acc_test = test(test_loader, model, use_best=False)
print("Test set performance - test_acc: {}, test_loss: {}\n".format(acc_test, loss_test))

"""
Save results
"""
with open(res_path+ "%s-%s-%s-%s-nr-0%s.pickle" % (mode, dataset, loss_name, noise_type, 
            str(int(noise_rate * 10))), 'wb') as f:
    pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                'epoch_acc_train': np.asarray(epoch_acc_train), 
                'epoch_loss_test': np.asarray(epoch_loss_test), 
                'epoch_acc_test': np.asarray(epoch_acc_test), 
                'idx': np.asarray(idx), 'X_temp': X_temp,
                'X_train': X_train, 'X_val': X_val,
                'y_temp': y_temp, 'y_train':y_train, 'y_val':y_val,
                'idx_train': np.asarray(idx_train),
                'idx_val': np.asarray(idx_val), 
                'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Pickle file saved: " + res_path+ "%s-%s-%s-%s-nr-0%s.pickle" % (mode, dataset, loss_name, 
                noise_type, str(int(noise_rate * 10))), "\n")

# Print the elapsed time
elapsed = time.time() - t_start
print("\nelapsed time: \n", elapsed)

"""
Plot results
"""
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
from torch.utils.tensorboard import SummaryWriter

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


class MNIST_MetaNN(nn.Module):
    def __init__(self):
        super(MNIST_MetaNN, self).__init__()

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

    def batchnorm(self, input, weight=None, bias=None, running_mean=None, 
                running_var=None, training=True, eps=1e-5, momentum=0.1):

        ''' momentum = 1 restricts stats to the current mini-batch '''
        # This hack only works when momentum is 1 and avoids needing to track running stats
        # by substituting dummy variables
        running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).cuda()
        running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).cuda()
        return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    
    def forward(self, x, weights=None):

        if weights == None:
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
        else:
            x = F.relu(F.conv2d(x, weights[0], weights[1]))
            x = F.MaxPool2d(x, kernel_size=2, stride=2)
            x = self.batch_norm(x, weight=weights[6], bias=weights[7], momentum=1)
            x = F.relu(F.conv2d(x, weights[2], weights[3]))
            x = F.MaxPool2d(x, kernel_size=2, stride=2)
            x = self.batch_norm(x, weight=weights[8], bias=weights[9], momentum=1)
            x = F.relu(F.conv2d(x, weights[4], weights[5]))
            x = self.batch_norm(x, weight=weights[10], bias=weights[11], momentum=1)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(F.linear(x, weights[12], weights[13]))
            x = F.relu(F.linear(x, weights[14], weights[15]))

        return x


    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def meta_mlnt_train(train_loader, model_tch, model_stud, first_order=False):

    """
    Code inspired from:

    https://github.com/vmikulik/maml-pytorch/blob/master/MAML-Sines.ipynb
    https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0
    """

    loss_train_tch = 0.
    acc_train_tch = 0.
    correct_tch = 0

    loss_train_stud = 0.
    acc_train_stud = 0.
    correct_stud = 0

    model_tch.train()
    model_stud.train()

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        if epoch == 1:
        # copy parameters from model_stud to model_tch
        ## model_tch.copy(model_stud)
            with torch.no_grad():
                for param_tch, param_stud in zip(model_tch.parameters(), model_stud.parameters()):
                    param_tch.copy_(param_stud)
        else:
            # update the model_tch with EMA of prev. & new
            # parameters of model_stud
            with torch.no_grad():
                for param_tch, param_stud in zip(model_tch.parameters(), model_stud.parameters()):
                    param_tch.mul_(gamma).add_(1-gamma, param_stud)

        y = y.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)

        output_stud = model_stud(x)
        output_tch = model_tch(x)
        output_tch.detach_()
        # loss_pass_1 = loss_fn(output_stud, y)

        optimizer.zero_grad()
        # loss_pass_1.mean().backward(retain_graph=True)

        loss_meta = 0.

        for _ in range(M):
            tmp_wts = []
            tmp_wts_2 = []

            idx_perm = torch.randperm(y.shape[0])
            y_perm = y.clone().detach()
            for i in range(rho):
                idx_nbr = idx_perm[i]
                x_tmp_ref = x[idx_nbr,:,:,:].clone().detach()
                x_tmp_ref.view(1,x_tmp_ref.shape[0])
                x_tmp_ref = x_tmp_ref.expand(y.shape[0], x.shape[0])
                dist = torch.sum((x_tmp_ref - x.reshape(x.shape[0],-1))**2, dim=1)
                _, neighbours = torch.topk(dist, num_neighbours+1, largest=False)
                y_perm[idx_nbr] = y[neighbours[random.randint(1, num_neighbours)]]
            
            if not first_order:
                tmp_wts = [w.clone() for w in model_stud.parameters()]
                loss_pass_task = loss_fn(model_stud(x, tmp_wts), y_perm)
                grads_task = torch.autograd.grad(loss_pass_task, tmp_wts)
                tmp_wts_2 = [w - task_lr * grad for w, grad in zip(tmp_wts, grads_task)]
            else:
                loss_pass_task = loss_fn(model_stud(x), y_perm)
                grads_task = torch.autograd.grad(loss_pass_task, model_stud.parameters(), 
                            create_graph=True, retain_graph=True, only_inputs=True)

                for g in grads_task:
                    g.detach_()
                    g.requires_grad = False
                tmp_wts_2 = [w - task_lr * grad for w, grad in zip(tmp_wts, grads_task)]

            loss_consistency = (1/M)* KL_loss(F.log_softmax(model_stud(x, tmp_wts_2), dim=1), F.softmax(output_tch, dim=1))
            loss_meta += loss_consistency

        grads_meta = torch.autograd.grad(loss_meta, model_stud.parameters())

        tmp_wts = [w.clone() for w in model_stud.parameters()]
        tmp_wts_2 = [w - meta_lr * grad for w, grad in zip(model_stud.parameters(), grads_meta)]
        output_stud = model_stud(x)
        loss_class = loss_fn(output_stud, y)

        optimizer.zero_grad()
        loss_class.mean().backward(retain_graph=True)
        optimizer.step()


        # # Take the final GD step after meta-task updates
        # for w, grad in zip(model_stud.parameters(), grads_meta):
        #     w.grad = grad
        # optimizer.step()

        # Compute the accuracy and loss  ::: STUDENT
        output = model_stud(x)
        loss_fin = loss_fn(output, y)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_train += loss_fin.mean().item() # .item() for scalars, .tolist() in general
        correct_stud += pred.eq(y.to(device)).sum().item()

        # Compute the accuracy and loss  ::: TEACHER
        output_tch = model_tch(x)
        loss_fin_tch = loss_fn(output_tch, y)
        pred_prob_tch = F.softmax(output_tch, dim=1)
        pred_tch = torch.argmax(pred_prob_tch, dim=1)
        loss_train_tch += loss_fin_tch.mean().item() # .item() for scalars, .tolist() in general
        correct_tch += pred_tch.eq(y.to(device)).sum().item()

        batch_cnt = batch_id + 1

        loss_train_agg[list(map(int, idx.tolist()))] = np.asarray(loss_fin.tolist())
        acc_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.eq(y.to(device)).tolist())

    loss_train_tch /= batch_cnt
    acc_train_tch = 100.*correct_tch/len(train_loader.dataset)

    loss_train_stud /= batch_cnt
    acc_train_stud = 100.*correct_stud/len(train_loader.dataset)

    return loss_train_tch, acc_train_tch, loss_train_stud, acc_train_stud, loss_train_agg, acc_train_agg

def test(data_loader, model, run):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):

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


"""
Configuration
"""

num_epoch = 50
batch_size = 128
random_state = 422
num_runs = 1 # 5
learning_rate = 2e-4

dataset = "mnist" # "cifar10" "cifar100"
num_class = 10
mode = "meta_mlnt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Meta_MLNT hyper-params
"""
KL_loss = nn.KLDivLoss(reduction='batchmean')
task_lr = 2e-2
meta_lr = 2e-3
M = 10 # [5, 10, 15]
num_neighbours = 10
gamma = 0.99 #[0.99, 0.999]
rho = int(0.5 * batch_size) # [0.1, 0.2, 0.3, 0.4, 0.5] * batch_size
# tau = 0.3 # to be used if MLNT is used for >=2 iterations


parser = argparse.ArgumentParser(description='Meta MLNT - CVPR 2019')
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
        model_tch = MNIST_MetaNN()
        model_stud = MNIST_MetaNN()
    elif dataset in ["cifar10", "cifar100"]:
        # model = torchvision.models.resnet18()
        pass

    model_tch = model_tch.to(device)
    model_stud = model_stud.to(device)
    print(model_stud)

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

        model_stud.load_state_dict(torch.load(chkpt_path + "%s-%s-cce-%s-nr-0%s-mdl-stud-wts-run-%s.pt"
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


    # optimizer = optim.Adam(model_stud.parameters(), learning_rate)
    optimizer = optim.SGD(model_stud.parameters(), lr = learning_rate, momentum=0.9)

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
    writer.add_graph(model_tch, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
    writer.add_graph(model_stud, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
    writer.close()


    epoch_loss_train_tch = []
    epoch_acc_train_tch = []
    epoch_loss_train_stud = []
    epoch_acc_train_stud = []

    epoch_loss_test_tch = []
    epoch_acc_test_tch = []
    epoch_loss_test_stud = []
    epoch_acc_test_stud = []

    epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))

    t_start = time.time()

    for epoch in range(num_epoch):

        #Training set performance
        loss_train_tch, acc_train_tch, loss_train_stud, acc_train_stud, loss_train_agg, acc_train_agg = meta_mlnt_train(train_loader,
                            model_tch, model_stud, first_order=False)
        writer.add_scalar('training_loss_teacher', loss_train_tch, epoch)
        writer.add_scalar('training_accuracy_teacher', acc_train_tch, epoch)
        writer.add_scalar('training_loss_student', loss_train_stud, epoch)
        writer.add_scalar('training_accuracy_student', acc_train_stud, epoch)
        writer.close()
        

        # Validation set performance
        loss_val_tch, acc_val_tch = test(val_loader, model_tch, run)
        loss_val_stud, acc_val_stud = test(val_loader, model_stud, run)

        #Test set performance
        loss_test_tch, acc_test_tch = test(test_loader, model_tch, run)
        loss_test_stud, acc_test_stud = test(test_loader, model_stud, run)

        writer.add_scalar('test_loss_tch', loss_test_tch, epoch)
        writer.add_scalar('test_accuracy_tch', acc_test_tch, epoch)
        writer.add_scalar('test_loss_stud', loss_test_stud, epoch)
        writer.add_scalar('test_accuracy_stud', acc_test_stud, epoch)
        writer.close()

        epoch_loss_train_tch.append(loss_train_tch)
        epoch_acc_train_tch.append(acc_train_tch)
        epoch_loss_train_stud.append(loss_train_stud)
        epoch_acc_train_stud.append(acc_train_stud)    

        epoch_loss_train_agg[:, epoch] = loss_train_agg
        epoch_acc_train_agg[:, epoch] = acc_train_agg

        epoch_loss_test_tch.append(loss_test_tch)
        epoch_acc_test_tch.append(acc_test_tch)
        epoch_loss_test_stud.append(loss_test_stud)
        epoch_acc_test_stud.append(acc_test_stud)


        
        # Learning Rate Scheduler Update
        lr_scheduler_1.step(loss_val_stud)
        ##lr_scheduler_3.step()
        ##lr_scheduler_2.step()

        # Update best_acc_val and sample_wts_fin
        if epoch == 1:
            best_acc_val = acc_val_stud

        if acc_val > best_acc_val:
            best_acc_val = acc_val_stud
            best_model_wts = copy.deepcopy(model_stud.state_dict())
            torch.save(model_stud.state_dict(), chkpt_path + chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-stud-wts-run-%s.pt" % (mode, dataset, 
                                loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
            print("Model weights updated...\n")
        
        print("Teacher::: Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                    optimizer.param_groups[0]['lr'], 
                                                    loss_train_tch, loss_val_tch, loss_test_tch, 
                                                    acc_train_tch, acc_val_tch, acc_test_tch))

        print("Student::: Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                    optimizer.param_groups[0]['lr'], 
                                                    loss_train_stud, loss_val_stud, loss_test_stud, 
                                                    acc_train_stud, acc_val_stud, acc_test_stud))


    loss_test_tch, acc_test_tch = test(test_loader, model_tch, run)
    loss_test_stud, acc_test_stud = test(test_loader, model_stud, run)

    print(f"Run: {run}::: Teacher::: Test set performance \
            - test_acc: {acc_test_tch}, test_loss: \
            {loss_test_tch}\n")
    print(f"Run: {run}::: Student::: Test set performance \
            - test_acc: {acc_test_stud}, test_loss: \
            {loss_test_stud}\n")

    if noise_rate > 0.:
        torch.save(model_tch.state_dict(), chkpt_path + 
                    "%s-%s-%s-%s-nr-0%s-mdl-tch-wts-run-%s.pt" % (
                    mode, dataset, loss_name, noise_type, 
                    str(int(noise_rate * 10)), str(run)))

        torch.save(model_tch.state_dict(), chkpt_path + 
                    "%s-%s-%s-%s-nr-0%s-mdl-stud-wts-run-%s.pt" % (
                    mode, dataset, loss_name, noise_type, 
                    str(int(noise_rate * 10)), str(run)))

    # model_tch.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-tch-wts-run-%s.pt"
    #                     % (mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
    # model_tch = model_tch.to(device)

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
                pickle.dump({'epoch_loss_train_tch': np.asarray(epoch_loss_train_tch), 
                            'epoch_acc_train_tch': np.asarray(epoch_acc_train_tch),
                            'epoch_loss_test_tch': np.asarray(epoch_loss_test_tch), 
                            'epoch_acc_test_tch': np.asarray(epoch_acc_test_tch),
                            'epoch_loss_train_stud': np.asarray(epoch_loss_train_stud), 
                            'epoch_acc_train_stud': np.asarray(epoch_acc_train_stud),
                            'epoch_loss_test_stud': np.asarray(epoch_loss_test_stud), 
                            'epoch_acc_test_stud': np.asarray(epoch_acc_test_stud), 
                            'epoch_loss_train_agg': epoch_loss_train_agg,
                            'epoch_acc_train_agg': epoch_acc_train_agg,
                            'idx_train': idx_train, 'idx_tr_clean_ref': idx_tr_clean_ref, 
                            'idx_tr_noisy_ref': idx_tr_noisy_ref,
                            'X_temp': X_temp,  'X_train': X_train, 
                            'X_val': X_val, 'y_temp': y_temp, 
                            'y_train':y_train, 'y_val':y_val,
                            'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump({'epoch_loss_train_tch': np.asarray(epoch_loss_train_tch), 
                #             'epoch_acc_train_tch': np.asarray(epoch_acc_train_tch),
                #             'epoch_loss_test_tch': np.asarray(epoch_loss_test_tch), 
                #             'epoch_acc_test_tch': np.asarray(epoch_acc_test_tch),
                #             'epoch_loss_train_stud': np.asarray(epoch_loss_train_stud), 
                #             'epoch_acc_train_stud': np.asarray(epoch_acc_train_stud),
                #             'epoch_loss_test_stud': np.asarray(epoch_loss_test_stud), 
                #             'epoch_acc_test_stud': np.asarray(epoch_acc_test_stud), 
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
                pickle.dump({'epoch_loss_train_tch': np.asarray(epoch_loss_train_tch), 
                            'epoch_acc_train_tch': np.asarray(epoch_acc_train_tch),
                            'epoch_loss_test_tch': np.asarray(epoch_loss_test_tch), 
                            'epoch_acc_test_tch': np.asarray(epoch_acc_test_tch),
                            'epoch_loss_train_stud': np.asarray(epoch_loss_train_stud), 
                            'epoch_acc_train_stud': np.asarray(epoch_acc_train_stud),
                            'epoch_loss_test_stud': np.asarray(epoch_loss_test_stud), 
                            'epoch_acc_test_stud': np.asarray(epoch_acc_test_stud), 
                            'epoch_loss_train_agg': epoch_loss_train_agg,
                            'epoch_acc_train_agg': epoch_acc_train_agg,
                            'X_temp': X_temp, 'X_train': X_train, 'X_val': X_val,
                            'y_temp': y_temp, 'y_train':y_train, 'y_val':y_val,
                            'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump({'epoch_loss_train_tch': np.asarray(epoch_loss_train_tch), 
                #             'epoch_acc_train_tch': np.asarray(epoch_acc_train_tch),
                #             'epoch_loss_test_tch': np.asarray(epoch_loss_test_tch), 
                #             'epoch_acc_test_tch': np.asarray(epoch_acc_test_tch),
                #             'epoch_loss_train_stud': np.asarray(epoch_loss_train_stud), 
                #             'epoch_acc_train_stud': np.asarray(epoch_acc_train_stud),
                #             'epoch_loss_test_stud': np.asarray(epoch_loss_test_stud), 
                #             'epoch_acc_test_stud': np.asarray(epoch_acc_test_stud), 
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
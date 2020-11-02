from __future__ import print_function, absolute_import

import os
import time
import math
import pickle
import copy
import argparse

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import *
from adversarial_attacks import *
from losses import *

import numpy_indexed as npi

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8

def hist_1d(a, num_class=10):
    return np.histogram(a, bins=num_class, range=(0,hist_lim-1))[0]

def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc

def selfie_train(epoch, ep_warm_up, hist_lim, uncertainty_thres, max_uncertainty,
             H, H_post_prob, H_uncertain, idx_refurb_fin, X_train, X_temp, 
             train_loader, dataset_train, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    idx_refurb_epoch = []
    idx_selfie_size = 0

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))

    model.train()

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)

        output = model(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_batch = loss_fn(output, y)
        idx_list = list(map(int, idx.tolist()))
        idx_selfie = idx

        if epoch > ep_warm_up:

            # pick (1-tau)*100 % low-loss valued samples
            _, idx_sort = torch.sort(loss_batch)
            if noise_rate > 0.:
                idx_low_loss = torch.index_select(idx_sort, 0, idx_sort[0:int((1.0- noise_rate) * x.shape[0])])
                idx_low_loss = idx_low_loss.type(torch.LongTensor)
            else:
                idx_low_loss = idx_sort.type(torch.LongTensor)


            # Choosing refurbishable sample and clean the labels

            ## find posterior prob. based on history (H) of predictions
            pred_freq = np.apply_along_axis(hist_1d, axis=1, arr=H[idx_list,:])
            pred_freq = torch.Tensor(pred_freq).type(torch.LongTensor)
        
            H_post_prob[idx_list, :] = (1/hist_lim*1.0) * (np.asarray(
                                        pred_freq.tolist())).astype(np.float32)

            ## Compute the predictive uncertainty            
            H_log_prob = np.log(H_post_prob[idx_list, :])
            H_log_prob[np.isinf(H_log_prob)] = 0.
            H_log_prob = H_log_prob.astype(np.float32)

            H_uncertain[idx_list] = np.sum(H_post_prob[idx_list,:] * -1 * H_log_prob, axis=1)

            H_uncertain[idx_list] *= 1./max_uncertainty
            

            if (H_uncertain[idx_list] <= uncertainty_thres).any():

                ## Select the refurbishable samples
                idx_refurb_tmp = idx[(H_uncertain[idx_list] 
                                        <= uncertainty_thres).tolist()]
                idx_refurb_tmp = idx_refurb_tmp.type(torch.LongTensor)
                idx_refurb_tmp_ref = npi.indices(X_train, X_temp[list(map(int, idx_refurb_tmp.tolist())),:,:,:]) 
                idx_refurb_batch = npi.indices(np.asarray(idx.tolist()), np.asarray(idx_refurb_tmp.tolist()))
                idx_refurb_batch = (torch.Tensor(idx_refurb_batch)).type(torch.LongTensor)
            

                # if batch_id == 0:
                #     print("\nBefore: y - \n {} \n".format(y[list(map(int,idx_refurb_batch.tolist()))]))
                ## clean the labels for the batch and in the actual dataset
                refurb_lab = torch.argmax(pred_freq[list(map(int, idx_refurb_batch.tolist())),:], dim=1)
                refurb_lab = refurb_lab.to(device)
                y[list(map(int,idx_refurb_batch.tolist()))] = refurb_lab
                y = y.type(torch.LongTensor)
                dataset_train.tensors[1][list(map(int,idx_refurb_tmp_ref.tolist()))] = refurb_lab.type(torch.FloatTensor)

                # if batch_id == 0:
                #     print("\nAfter: y - \n {} \n".format(y[list(map(int,idx_refurb_batch.tolist()))]))

                ## store the indices of refurbished samples for 
                ## algorithm comparison and analysis 
                if len(idx_refurb_epoch) == 0:
                    idx_refurb_epoch = idx_refurb_tmp
                else:
                    idx_refurb_epoch = torch.cat((idx_refurb_epoch.to(device), idx_refurb_tmp.to(device)), dim=0)
                    idx_refurb_epoch = idx_refurb_epoch.type(torch.LongTensor)

                ## low-loss valued + refurbished samples
                idx_selfie = torch.unique(torch.cat((idx_low_loss.to(device), idx_refurb_batch.to(device)), dim=0))

                # if batch_id == 0:
                #     print("\n Batch {} \n idx_refurb_tmp_ref ({}): \n{}\n---\n".format(batch_id, idx_refurb_tmp_ref.shape, idx_refurb_tmp_ref))
                #     print("\n idx_sort_frac ({}): \n{}\n---\n".format(idx_sort[0:int((1.0- noise_rate) * x.shape[0])].shape, 
                #                                                 idx_sort[0:int((1.0- noise_rate) * x.shape[0])]))
                #     print("\n idx_sort ({}): \n{}\n---\n".format(idx_sort.shape, idx_sort))
                #     print("\n idx_low_loss ({}): \n{}\n---\n".format(idx_low_loss.shape, idx_low_loss))
                #     print("\n pred_freq ({}): \n{}\n---\n".format(pred_freq[0:15].shape, pred_freq[0:15]))
                #     print("\n H_post_prob:\n {}\n---\n".format(H_post_prob[list(map(int, idx[0:15].tolist())),:]))
                #     print("\n H_log_prob:\n {}\n---\n".format(H_log_prob[0:15,:]))
                #     print("\n H_uncertain:\n {}\n---\n".format(H_uncertain[list(map(int, idx[0:15].tolist()))]))
                #     print("\n idx_refurb_tmp ({}): \n{}\n---\n".format(idx_refurb_tmp.shape, idx_refurb_tmp))
                #     print("\n idx_refurb_batch ({}): \n{}\n---\n".format(idx_refurb_batch.shape, idx_refurb_batch))
                #     print("\n idx_selfie ({}): \n{}\n---\n".format(idx_selfie.shape, idx_selfie))
                #     print("\n idx_refurb_epoch ({}): \n{}\n---\n".format(idx_refurb_epoch.shape, idx_refurb_epoch))
                #     input("Press <ENTER> to continue...\n")

            else:
                ## Picking only the low-loss valued samples
                idx_selfie = idx_low_loss


            # Also select the previously refurbished samples if they 
            # are present in this batch
            if (epoch > ep_warm_up + 1) and ((H_uncertain[idx_list] 
                                > uncertainty_thres).any()) and len(idx_refurb_fin) != 0:
                                
                idx_tmp = idx[(H_uncertain[idx_list] > uncertainty_thres).tolist()]

                # Check which of idx_tmp entries are in idx_refurb_fin
                diff_list_tmp = list(set(list(map(int,idx_refurb_fin.tolist()))).intersection(set(list(map(int,idx_tmp.tolist())))))
                diff_list_batch = npi.indices(np.asarray(idx_list), np.asarray(diff_list_tmp))
                idx_refurb_old = (torch.Tensor(diff_list_batch)).type(torch.LongTensor)    
                idx_selfie = torch.unique(torch.cat((idx_selfie, idx_refurb_old.to(device)), dim=0))
                    

        if epoch > ep_warm_up:
            # loss_selfie = torch.index_select(loss_batch, 0, (idx_selfie.type(torch.LongTensor)).to(device))
            output_selfie = model(torch.index_select(x, 0, (idx_selfie.type(torch.LongTensor)).to(device)))
            y_selfie = torch.index_select(y.to(device), 0, (idx_selfie.type(torch.LongTensor)).to(device))
            loss_selfie = loss_fn(output_selfie, y_selfie)
        else:
            loss_selfie = loss_batch

        optimizer.zero_grad()
        loss_selfie.mean().backward()
        optimizer.step()

        # Final class predictions after "refurbishment"
        output_fin = model(x)
        pred_selfie = torch.argmax(F.softmax(output_fin, dim=1), dim=1)
        loss_train += (torch.mean(loss_fn(output_fin, y.to(device)))).item() # .item() for scalars, .tolist() in general
        correct += (pred_selfie.eq(y.to(device))).sum().item()

        # Update the History (look-up) table
        if epoch > ep_warm_up - hist_lim:
            # pred = torch.argmax(F.softmax(model(x), dim=1), dim=1)
            H[idx_list, (epoch - (ep_warm_up - hist_lim) - 1) % hist_lim] = list(map(int,pred_selfie.tolist()))


        batch_cnt = batch_id + 1

        idx_selfie_size += idx_selfie.shape[0]

        loss_train_agg[list(map(int, idx_sort.tolist()))] = np.asarray(loss_batch.tolist())
        acc_train_agg[list(map(int, idx_sort.tolist()))] = np.asarray(pred.eq(y.to(device)).tolist())

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return idx_refurb_epoch, loss_train, acc_train, idx_selfie_size, loss_train_agg, acc_train_agg

def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load("%s-%s-eps-00%s-q-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, 
                                                    str(int(uncertainty_thres*100)), str(hist_lim), loss_name, noise_type, 
                                                    str(int(noise_rate * 10)), str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)
        
            x, y = x.to(device), y.to(device)

            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_batch = loss_fn(output, y)
            # loss_batch = loss_fn(output, y_true)

            loss_test += torch.mean(loss_batch).item()
            # loss_test += loss_batch.item() # - for L_DMI
            correct += (pred.eq(y.to(device))).sum().item()

            batch_cnt = batch_id + 1
        
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)

    return loss_test, acc_test


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

class mem_NN(nn.Module):

    def __init__(self):
        super(mem_NN, self).__init__()

        # self.fc1 = nn.Linear(2, 512)
        # self.fc2 = nn.Linear(512, 64)
        # self.fc3 = nn.Linear(64, 16)
        # self.fc4 = nn.Linear(16, 8)
        # self.fc5 = nn.Linear(8, 4)

        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)

        self.dr1 = nn.Dropout(0.2)
        # self.dr2 = nn.Dropout(0.2)
        # self.dr3 = nn.Dropout(0.2)

        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(16)
        # self.bn4 = nn.BatchNorm1d(8)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        # x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        # x = self.dr2(x)
        ## x = F.relu(self.fc3(x))
        ## x = self.bn3(x)
        # x = self.dr3(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = F.relu(self.fc5(x))
        return x


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


"""
Configuration
"""

num_epoch = 200
batch_size = 128
num_runs = 1 # 5
learning_rate = 2e-4

parser = argparse.ArgumentParser(description='SELFIE - ICML 2019')
parser.add_argument('-nr','--noise_rate', default=0.6, type=float, help='noise_rate')
parser.add_argument('-nt','--noise_type', default="sym", type=str, help='noise_type')
parser.add_argument('-loss','--loss_name', default="cce", type=str, help='loss_name')
args = parser.parse_args()

noise_rate = args.noise_rate
noise_type = noise_type = args.noise_type
loss_name = args.loss_name # "cce" # "mae" # "dmi" # "cce" # "rll"

dataset = "mnist" # "checker_board" # "board" #  "cifar10" "cifar100"
mode = "selfie"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# selfie_params = {'ep_warm_up': 25, 'hist_lim': 15, 'uncertainty_thres': 0.05, 
#                'max_uncertainty': math.log(num_class)}

for run in range(num_runs):

    print("\n==================\n")
    print(f"== RUN No.: {run} ==")
    print("\n==================\n")

    for uncertainty_thres in [0.05]:#, 0.1, 0.15, 0.2]:

        for hist_lim in [10]:#, 15, 20]:

            chkpt_path = "./checkpoint/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/q-" + str(hist_lim) + "/eps-" + str(int(100 * uncertainty_thres)) + "/run_" + str(run) + "/"

            res_path = "./results_pkl/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/q-" + str(hist_lim) + "/eps-" + str(int(100 * uncertainty_thres)) + "/run_" + str(run) + "/"

            plt_path = "./plots/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/q-" + str(hist_lim) + "/eps-" + str(int(100 * uncertainty_thres)) + "/run_" + str(run) + "/"

            log_dirs_path = "./runs/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/q-" + str(hist_lim) + "/eps-" + str(int(100 * uncertainty_thres)) + "/run_" + str(run) + "/"

            if not os.path.exists(chkpt_path):
                os.makedirs(chkpt_path)

            if not os.path.exists(res_path):
                os.makedirs(res_path)

            if not os.path.exists(plt_path):
                os.makedirs(plt_path)

            if not os.path.exists(log_dirs_path):
                os.makedirs(log_dirs_path)

            t_start = time.time()

            #History (look-up) table
            # H = np.empty(shape=(X_train.shape[0], hist_lim), dtype=np.float32)
            H = np.empty(shape=(X_temp.shape[0], hist_lim), dtype=np.float32)

            #Posterior Prob. based on History
            H_post_prob = np.empty(shape=(X_temp.shape[0], num_class), dtype=np.float32)

            # Predictive uncertainty table
            # H_uncertain = np.empty(shape=(X_train.shape[0], ), dtype=np.float32)
            H_uncertain = np.empty(shape=(X_temp.shape[0], ), dtype=np.float32)

            # Indices of refurbished samples
            idx_refurb_fin = []

            """
            Read DATA
            """

            if dataset in ["board", "checker_board", "mnist", "cifar10", "cifar100"]:
                dat, ids = read_data(noise_type, noise_rate, dataset)

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

            ep_warm_up = 25
            max_uncertainty = math.log(num_class)


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
            Choose MODEL and LOSS FUNCTION
            """
            
            if dataset in ["board", "checker_board"]:
                model = mem_NN()
            elif dataset == "mnist":
                model = MNIST_CNN()
                # model = MNIST_ResNet18()
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
                
                model.load_state_dict(torch.load(chkpt_path + "%s-%s-eps-00%s-q-%s-cce-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, 
                                                    str(int(uncertainty_thres*100)), str(hist_lim), noise_type, 
                                                    str(int(noise_rate * 10)), str(run))))
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


            # optimizer = optim.Adam(model.parameters(), learning_rate)
            # optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.)
            optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay = 5e-4)

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

            epoch_loss_train = []
            epoch_acc_train = []
            epoch_loss_test = []
            epoch_acc_test = []

            idx_selfie_size_list = []

            best_acc_val = 0.

            """
            Aggregate sample-wise loss values for each epoch
            """
            epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
            epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))

            for epoch in range(num_epoch):


                #Training set performance
                idx_refurb_temp, loss_train, acc_train, idx_selfie_size, loss_train_agg, acc_train_agg = selfie_train(epoch, ep_warm_up, hist_lim, 
                                    uncertainty_thres, max_uncertainty, H, H_post_prob, 
                                    H_uncertain, idx_refurb_fin, X_train, X_temp, train_loader, dataset_train, model)

                idx_selfie_size_list.append(idx_selfie_size/tensor_x_train.shape[0])

                # Validation set performance
                loss_val, acc_val = test(val_loader, model, run, use_best=False)
                #Testing set performance
                loss_test, acc_test = test(test_loader, model, run, use_best=False)

                # Learning Rate Scheduler Update
                lr_scheduler_1.step(loss_val)
                ##lr_scheduler_3.step()
                ##lr_scheduler_2.step()

                epoch_loss_train.append(loss_train)
                epoch_acc_train.append(acc_train)
                epoch_loss_train_agg[:, epoch] = loss_train_agg
                epoch_acc_train_agg[:, epoch] = acc_train_agg    

                epoch_loss_test.append(loss_test)
                epoch_acc_test.append(acc_test)

                # Update best_acc_val
                if epoch == 0:
                    best_acc_val = acc_val

                # Update list of refurbished samples' indices
                if len(idx_refurb_fin) == 0 and len(idx_refurb_temp) != 0:
                    idx_refurb_fin = idx_refurb_temp.type(torch.FloatTensor)
                    idx_refurb_fin = torch.Tensor(idx_refurb_fin)
                if len(idx_refurb_fin) != 0 and len(idx_refurb_temp) != 0:
                    idx_refurb_fin = torch.cat((idx_refurb_fin, idx_refurb_temp.type(torch.FloatTensor)), dim=0)

                if (acc_val > best_acc_val and epoch > ep_warm_up) or epoch == ep_warm_up + 1:
                    best_acc_val = acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), chkpt_path + "%s-%s-eps-00%s-q-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, 
                                                    str(int(uncertainty_thres*100)), str(hist_lim), loss_name, noise_type, 
                                                    str(int(noise_rate * 10)), str(run)))
                    print("Model weights updated...\n")
                    best_acc_val = acc_val

                print("idx_selfie_size: {} - {}\n".format(idx_selfie_size_list[epoch-1], idx_selfie_size/tensor_x_train.shape[0]))
                print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                    optimizer.param_groups[0]['lr'], 
                                                    loss_train, loss_val, loss_test, 
                                                    acc_train, acc_val, acc_test))


            # Check & remove the duplicate entries in idx_refurb_fin
            idx_refurb_fin = list(set(idx_refurb_fin))

            # Test accuracy
            loss_test, acc_test = test(test_loader, model, run, use_best=False)

            print("\n uncertainty_thres: {}, hist_lim: {}\n".format(uncertainty_thres, hist_lim))
            print(f"Run: {run}::: Test set performance - test_acc: {acc_test}, test_loss: {loss_test}\n")

            if noise_rate > 0.:
                torch.save(model.state_dict(), chkpt_path + "%s-%s-eps-00%s-q-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, 
                                                    str(int(uncertainty_thres*100)), str(hist_lim), loss_name, noise_type, 
                                                    str(int(noise_rate * 10)), str(run)))

                

            # model.load_state_dict(torch.load(chkpt_path + "%s-%s-eps-00%s-q-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, 
            #                                        str(int(uncertainty_thres*100)), str(hist_lim), loss_name, noise_type, 
            #                                        str(int(noise_rate * 10)), str(run)))
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
            with open(res_path+ "%s-%s-eps-00%s-q-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, 
                                str(int(uncertainty_thres*100)), str(hist_lim), loss_name, noise_type, 
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

            print("Pickle file saved: " + res_path + "%s-%s-eps-00%s-q-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, 
                                str(int(uncertainty_thres*100)), str(hist_lim), loss_name, noise_type, 
                                str(int(noise_rate * 10)), str(run)), "\n")

            # Print the elapsed time
            elapsed = time.time() - t_start
            print("\nelapsed time: \n", elapsed)


            """
            Plot results and save FIG
            """

            # plot_selfie(dataset, noise_rate, noise_type, run, hist_lim, uncertainty_thres)

"""
# correction log

if method == "selfie":
    num_corrected_sample = 0
    num_correct_corrected_sample = 0

    samples = train_batch_patcher.loaded_data
    for sample in samples:
        if sample.corrected:
            num_corrected_sample += 1
            if sample.true_label == sample.last_corrected_label:
                num_correct_corrected_sample += 1

    if num_corrected_sample != 0:
        print("Label correction of ""refurbishable"" samples : ",(epoch + cur_epoch + 1), ": ", num_correct_corrected_sample, "/", num_corrected_sample, "( ", float(num_correct_corrected_sample)/float(num_corrected_sample), ")")
        if correction_log is not None:
            correction_log.append(str(epoch + cur_epoch + 1) + ", " + str(num_correct_corrected_sample) + ", " + str(num_corrected_sample) + ", " + str(float(num_correct_corrected_sample)/float(num_corrected_sample)))

"""


"""
You can compute the F-score yourself in pytorch. The F1-score is defined for single-class (true/false) 
classification only. The only thing you need is to aggregating the number of:
      Count of the class in the ground truth target data;
      Count of the class in the predictions;
      Count how many times the class was correctly predicted.

Let's assume you want to compute F1 score for the class with index 0 in your softmax. In every batch, you can do:

predicted_classes = torch.argmax(y_pred, dim=1) == 0
target_classes = self.get_vector(y_batch)
target_true += torch.sum(target_classes == 0).float()
predicted_true += torch.sum(predicted_classes).float()
correct_true += torch.sum(
    predicted_classes == target_classes * predicted_classes == 0).float()

When all batches are processed:

recall = correct_true / target_true
precision = correct_true / predicted_true
f1_score = 2 * precission * recall / (precision + recall)

"""
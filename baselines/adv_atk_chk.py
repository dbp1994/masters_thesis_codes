from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import pickle
import copy
from tqdm import tqdm
import argparse

from sklearn import model_selection

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import *
from adversarial_attacks import *

from losses import *


# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8

# Model
class MNIST_NN(nn.Module):
    def __init__(self):
        super(MNIST_NN, self).__init__()

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


if __name__ == "__main__":

    num_epoch = 200
    num_runs = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    parser = argparse.ArgumentParser(description='PyTorch "Memorization Check in NNs" Training')
    parser.add_argument('--nr', default=0.25, type=float, help='noise_rate')
    parser.add_argument('--nt', default="sym", type=str, help='noise_type')
    parser.add_argument('--ln', default="cce", type=str, help='loss_name')
    args = parser.parse_args()


    noise_rate = 0.6 # noise_rate = args.nr
    noise_type = "sym" # noise_type = args.nt

    dataset = "mnist" # "checker_board" # "board" #  "mnist" # "board"
    mode = "adv_atk"

    for run in range(num_runs):


        print("\n==================\n")
        print(f"== RUN No.: {run} ==")
        print("\n==================\n")

        t_start = time.time()

        chkpt_path = "./checkpoint/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/" + "run_" + str(run) + "/"

        res_path = "./results_pkl/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/" + "run_" + str(run) + "/"

        plt_path = "./plots/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/" + "run_" + str(run) + "/"

        log_dirs_path = "./runs/" + mode + "/" + dataset + "/" + noise_type + "/0" + str(int(noise_rate*10)) + "/" + "run_" + str(run) + "/"

        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)

        if not os.path.exists(res_path):
            os.makedirs(res_path)

        if not os.path.exists(plt_path):
            os.makedirs(plt_path)

        if not os.path.exists(log_dirs_path):
            os.makedirs(log_dirs_path)

        """
        Read DATA
        Train -- Val -- Test SPLIT
        """

        if dataset in ["board", "checker_board"]:
            dat = np.loadtxt(f"./{dataset}/{dataset}_data.txt", delimiter=",")
            X, y = dat[:,:-1], dat[:,-1]

            if int(np.min(y)) == 0:
                num_class = int(np.max(y) + 1)
            else:
                num_class = int(np.max(y))

            ## training data - use only X % of available data
            # use_frac = 0.05
            # X1, _, y1, _ = model_selection.train_test_split(X, y, 
            #                     test_size= 1. - use_frac, random_state=42)

            # X_temp, X_test, y_temp, y_test = model_selection.train_test_split(X1, y1, 
            #                     test_size=0.2, random_state=42)
            
            X_temp, X_test, y_temp, y_test = model_selection.train_test_split(X, y, 
                                test_size=0.2, random_state=42)

        elif dataset == "mnist":
            dat, _ = read_data(noise_type, noise_rate, dataset)
            X, y = dat[0], dat[1]
            X_test, y_test = dat[6], dat[7]

            if int(np.min(y)) == 0:
                num_class = int(np.max(y) + 1)
            else:
                num_class = int(np.max(y))

        else:
            raise SystemExit("Dataset not supported.\n")

        
        # Test set
        tensor_x_test = torch.Tensor(X_test)
        tensor_y_test = torch.Tensor(y_test)
        test_size = 100
        dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_size, shuffle=True)


        """
        Choose MODEL and LOSS FUNCTION
        """
        
        if dataset in ["board", "checker_board"]:
            model = mem_NN()
        elif dataset == "mnist":
            model = MNIST_NN()

        model = model.to(device)
        print(model)

        
        loss_name = args.ln # "cce" # "mae" # "dmi" # "cce" # "rll"
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
            model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-cce-nr-0%s-mdl-wts-run-%s.pt"
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
        

        """
        Load the saved model
        """
        model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt"
                            % (mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
        model = model.to(device)

        """
        Adversarial Attacks
        """
        if dataset=="mnist":
            epsilon = 0.1
            k = 40
            # alpha = 2.5 * epsilon / k
        
        adv_acc_fgsm = test_fgsm_batch_ce(model, test_loader, epsilon)
        adv_acc_pgd = test_pgd_batch_ce(model, test_loader, epsilon, k=k)
        # adv_acc_df = test_df_batch_ce(model, test_loader, epsilon)

        print("Checking performance for FGSM & PGD Adversarial attacks...\n")
        print(f"Run {run} ::: FGSM Attack: acc. = {adv_acc_fgsm}, PGD Attack: acc. = {adv_acc_pgd}\n")

        """
        Save results
        """
        with open(res_path+ "%s-%s-%s-%s-nr-0%s-adv-atk-run-%s.pickle" % (mode, dataset, loss_name, noise_type, 
                    str(int(noise_rate * 10)), str(run)), 'wb') as f:
                pickle.dump({'num_epoch': num_epoch, 
                            'adv_acc_fgsm': adv_acc_fgsm, 
                            'adv_acc_pgd': adv_acc_pgd}, f, 
                            protocol=pickle.HIGHEST_PROTOCOL)

        print("Pickle file saved: " + res_path+ "%s-%s-%s-%s-nr-0%s-adv-atk-run-%s.pickle" % (mode, dataset, loss_name, 
                        noise_type, str(int(noise_rate * 10)), str(run)), "\n")

        # Print the elapsed time
        elapsed = time.time() - t_start
        print("\nelapsed time: \n", elapsed)
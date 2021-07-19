from __future__ import print_function, absolute_import

import numpy as np
import os
import time
import math
import pickle
import copy
from tqdm import tqdm
import argparse
import datetime
import distutils
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet, BasicBlock
from pakdd_models import CIFAR_CNN, CIFAR_Inception_Small, CIFAR_AlexCNN_Small, CIFAR_MLP_3x, CIFAR_MLP_1x, CIFAR_ResNet34, CIFAR_ResNet32
from resnet import resnet32

from data import *
from adversarial_attacks import *
from losses import get_loss

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8

class MNIST_ResNet18(ResNet):
    def __init__(self):
        super(MNIST_ResNet18, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

class MNIST_CNN_2(nn.Module):
    def __init__(self):
        super(MNIST_CNN_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
            
        self.lin1 = nn.Linear(200, 512)
        self.lin2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.bn1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.bn2(x)
        x = self.bn3(self.pool3(F.relu(self.conv5(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

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


def train(train_loader, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    model.train()

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))
    pred_train_agg = np.zeros((len(train_loader.dataset), ))

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
        correct += (pred.eq(y)).sum().item()

        batch_cnt = batch_id + 1

        loss_train_agg[list(map(int, idx.tolist()))] = np.asarray(loss_batch.tolist())
        acc_train_agg[list(map(int, idx.tolist()))] = np.asarray((pred.eq(y)).tolist())
        pred_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.tolist())

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return loss_train, acc_train, loss_train_agg, acc_train_agg, pred_train_agg


def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    acc_test = 0.
    correct = 0

    model.eval()

    print("Testing...\n")

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load(chkpt_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                                            mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                                            loss_name, noise_type, str(int(noise_rate * 10)), str(run)))['model_state_dict'])
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch 'Memorization Check in NNs' Training")
    parser.add_argument("-dat", "--dataset", default="cifar10", type=str, help="dataset")
    parser.add_argument("-nr","--noise_rate", default=0.4, type=float, help="noise_rate")
    parser.add_argument("-nt","--noise_type", default="sym", type=str, help="noise_type")
    parser.add_argument("-loss","--loss_name", default="cce", type=str, help="loss_name")
    parser.add_argument("-arch", "--architecture", default="resnet", type=str, help="architecture")
    parser.add_argument("-bn", "--batch_norm", default=1, type=int, help="Batch Normalization", choices=[0, 1])
    parser.add_argument("-wd", "--weight_decay", default=1e-4, type=float, help="weight decay for optimizer")
    parser.add_argument("-da", "--data_aug", default=1, type=int, help="data augmentation", choices=[0, 1])
    parser.add_argument("-gpu", "--gpu_id", default='0', type=str, help="GPU_ID: ['0', '1']")
    # parser.add_argument("--bn", dest="bn", type=lambda x:bool(distutils.util.strtobool(x)), help="Batch Normalization")
    # parser.add_argument("--da", dest="da", action='store_false', help="data augmentation")

    args = parser.parse_args()

    dataset = args.dataset # "cifar10"
    noise_rate = args.noise_rate # 0.6
    noise_type = args.noise_type # "sym"
    loss_name = args.loss_name # "cce" # "mae" # "dmi" # "cce" # "rll"
    arch = args.architecture # "inception"
    batch_norm = bool(args.batch_norm)
    weight_decay = args.weight_decay
    data_aug = bool(args.data_aug)
    gpu_id = args.gpu_id


    num_runs = 100
    batch_size = 128

    print(f"batch_norm: {batch_norm}, weight_decay: {weight_decay}\n")

    device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else "cpu")
    mode = "loss_reg_pat"

    t_start = time.time()

    for run in range(num_runs):

        print("\n==================\n")
        print(f"== Run. No.: {run} ==")
        print("\n==================\n")

        chkpt_path = f"./checkpoint/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        res_path = f"./results_pkl/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        log_dirs_path = f"./runs/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)

        if not os.path.exists(res_path):
            os.makedirs(res_path)

        if not os.path.exists(log_dirs_path):
            os.makedirs(log_dirs_path)
        else:
            for f in pathlib.Path(log_dirs_path).glob('events.out*'):
                try:
                    f.unlink()
                except OSError as e:
                    print(f"Error: {f} : {e.strerror}")
        

        print("\n============ PATHS =================\n")
        print(f"chkpt_path: {chkpt_path}")
        print(f"res_path: {res_path}")
        print(f"log_dirs_path: {log_dirs_path}")
        print("file name: %s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                    mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                    loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
        print("\n=============================\n")


        """
        Read DATA
        """

        dat, ids = read_data(noise_type, noise_rate, dataset, data_aug, mode)

        y_temp = dat[1]
        X_train, y_train = dat[2], dat[3]

        if arch in ["mlp1x", "mlp3x"]:
            X_train = X_train.reshape(X_train.shape[0], -1)

        idx_train = ids[1]

        if int(np.min(y_train)) == 0:
            num_class = int(np.max(y_train) + 1)
        else:
            num_class = int(np.max(y_train))

        print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))

        tensor_x_train = torch.Tensor(X_train) 
        tensor_y_train = torch.Tensor(y_train)
        tensor_id_train = torch.from_numpy(np.asarray(list(range(X_train.shape[0]))))

        dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train, tensor_id_train)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        """
        Choose MODEL and LOSS FUNCTION
        """
        if arch == "mlp1x":
            model = CIFAR_MLP_1x(X_train.shape[1], num_class)
            learning_rate = 1e-2
        elif arch == "mlp3x":
            model = CIFAR_MLP_3x(X_train.shape[1], num_class)
            learning_rate = 1e-2
        elif arch == "cnn":
            model = CIFAR_CNN(num_class, batch_norm=batch_norm)
            learning_rate = 1e-3
        elif arch == "resnet":
            # model = CIFAR_ResNet34(num_class)
            # model = CIFAR_ResNet32(num_class)
            model = resnet32()
            learning_rate = 1e-1
        elif arch == "inception_small":
            model = CIFAR_Inception_Small(num_classes=num_class, batch_norm=batch_norm)
            learning_rate = 1e-1
            # model = torchvision.models.googlenet(pretrained=False, num_classes=num_class, aux_logits=False)
        elif arch == "alexnet":
            model = CIFAR_AlexCNN_Small(num_classes=num_class, batch_norm=batch_norm)
            learning_rate = 1e-2
            # model = torchvision.models.alexnet(pretrained=False, num_classes=num_class)
        elif arch == "inception_v3":
            model = torchvision.models.inception_v3(pretrained=False, num_classes=num_class, aux_logits=False)
            learning_rate = 1e-1
        
        # model = MNIST_CNN()
        # learning_rate = 2e-4
        # model = MNIST_CNN_2()
        # learning_rate = 1e-3 - ADAM
        # # model = MNIST_ResNet18()
        # learning_rate = 2e-4

        model = model.to(device)

        if run == 0:
            print(model)
    
        kwargs = {}

        if loss_name == "rll":
            kwargs['alpha'] = 0.01 #  0.45 # 0.1
        elif loss_name == "gce":
            kwargs['q'] = 0.7
        elif loss_name == "norm_mse":
            kwargs['alpha'] = 0.1
            kwargs['beta'] = 1.

        loss_fn = get_loss(loss_name, num_class, reduction="none", **kwargs)
        
        print("\n===========\nloss: {}\n===========\n".format(loss_name))

        if loss_name == "dmi":
            model.load_state_dict(torch.load(chkpt_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                                        mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                                        loss_name, noise_type, str(int(noise_rate * 10)), str(run)))['model_state_dict'])

        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        if run == 0:
            exp_loss_train_agg = np.zeros((len(train_loader.dataset), num_runs))
            exp_acc_train_agg = np.zeros((len(train_loader.dataset), num_runs))
            exp_pred_train_agg = np.zeros((len(train_loader.dataset), num_runs))

        # Training
        loss_train, acc_train, loss_train_agg, acc_train_agg, pred_train_agg = train(train_loader, model)

        exp_loss_train_agg[:, run] = np.asarray(loss_train_agg)
        exp_acc_train_agg[:, run] = np.asarray(acc_train_agg)
        exp_pred_train_agg[:, run] = np.asarray(pred_train_agg)

        state_dict = {'model_state_dict': model.state_dict(), 
                    'opt_state_dict': optimizer.state_dict(),
                    'run': run
                    }
        torch.save(state_dict, chkpt_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                                mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                                loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
        

        print("Run: {}, lr: {}, loss_train: {}, acc_train: {}\n".format(run, 
                                            optimizer.param_groups[0]['lr'], 
                                            loss_train, acc_train))
        
        """
        Save results
        """
        with open(res_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (
                            mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                            loss_name, noise_type, 
                            str(int(noise_rate * 10)), 
                            str(run)), 'wb') as f:
            pickle.dump({'exp_acc_train_agg': exp_acc_train_agg,
                        'exp_pred_train_agg': exp_pred_train_agg,
                        'y_train_org': y_temp[idx_train], 'y_train':y_train,
                        'num_runs': num_runs}, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Pickle file saved: " + res_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (
                    mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                    loss_name, noise_type, 
                    str(int(noise_rate * 10)), 
                    str(run)), "\n")

        log_txt_name = "%s-%s-log.txt" % (mode, dataset)
        file_name = "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.[pickle/pt]" % (
                    mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                    loss_name, noise_type, str(int(noise_rate * 10)), str(run))

        with open(log_txt_name, 'a') as f:
            f.write("\n===================================================\n")
            f.write(f"file_name: {file_name}\n")
            f.write(f"Date & Time: {str(datetime.datetime.now())}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"noise_rate: {noise_rate}\n")
            f.write(f"noise_type: {noise_type}\n")
            f.write(f"loss: {loss_name}\n")
            f.write(f"mode: {mode}\n")
            f.write(f"arch: {arch}\n")
            f.write(f"batch_norm: {batch_norm}\n")
            f.write(f"weight_decay: {weight_decay}\n")
            f.write(f"data_aug: {data_aug}\n")
            f.write(f"num_runs: {num_runs}\n")
            f.write(f"batch_size: {batch_size}\n")

    # Print the elapsed time
    elapsed = time.time() - t_start
    print("\nelapsed time: \n", elapsed)

    with open(log_txt_name, 'a') as f:
        f.write(f"elapsed_time: {elapsed}")
        f.write("\n===================================================\n")
        
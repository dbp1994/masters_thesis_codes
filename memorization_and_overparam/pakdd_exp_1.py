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

import numpy_indexed as npi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet, BasicBlock
from pakdd_models import CIFAR_Inception_Small
from resnet import resnet32

from data import *
from losses import get_loss


# set seed for reproducibility
# torch.manual_seed(1337)
# np.random.seed(3459)
torch.manual_seed(333)
np.random.seed(4578)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8

class MNIST_ResNet18(ResNet):
    def __init__(self):
        super(MNIST_ResNet18, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

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
    parser.add_argument("-arch", "--architecture", default="inception_small", type=str, help="architecture")
    parser.add_argument("-bn", "--batch_norm", default=1, type=int, help="Batch Normalization", choices=[0, 1])
    parser.add_argument("-wd", "--weight_decay", default=0., type=float, help="weight decay for optimizer")
    parser.add_argument("-da", "--data_aug", default=1, type=int, help="data augmentation", choices=[0, 1])
    parser.add_argument("-ep", "--num_epoch", default=200, type=int, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("-run", "--num_runs", default=1, type=int, help="number of runs")
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
    num_epoch = args.num_epoch
    batch_size = args.batch_size # 128
    num_runs = args.num_runs # 1
    gpu_id = args.gpu_id

    print(f"batch_norm: {batch_norm}, weight_decay: {weight_decay}\n")

    device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else "cpu")
    mode = "loss_reg"

    for run in range(num_runs):

        epoch_loss_train = []
        epoch_acc_train = []

        epoch_loss_test = []
        epoch_acc_test = []

        print("\n==================\n")
        print(f"== RUN No.: {run} ==")
        print("\n==================\n")

        t_start = time.time()

        chkpt_path = f"./checkpoint/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        res_path = f"./results_pkl/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        plt_path = f"./plots/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        log_dirs_path = f"./runs/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

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
        print("file name: %s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                    mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                    loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
        print("\n=============================\n")


        """
        Read DATA
        """

        dat, ids = read_data(noise_type, noise_rate, dataset, data_aug)

        X_temp, y_temp = dat[0], dat[1]
        X_train, y_train = dat[2], dat[3]
        X_val, y_val = dat[4], dat[5]
        X_test, y_test = dat[6], dat[7]

        if arch in ["mlp1x", "mlp3x"]:
            X_temp = X_temp.reshape(X_temp.shape[0], -1)
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

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


        print("\n=============================\n")
        print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
        print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
        print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")
        print("y_train - min : {}, y_val - min : {}, y_test - min : {}".format(np.min(y_train), np.min(y_val), np.min(y_test)))
        print("y_train - max : {}, y_val - max : {}, y_test - max : {}".format(np.max(y_train), np.max(y_val), np.max(y_test)))
        print("\n=============================\n")
        print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))

        tensor_x_train = torch.Tensor(X_train) 
        tensor_y_train = torch.Tensor(y_train)
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


        print("\n tensor_x_train[0].unsqueeze(1): {}, {}\n".format(
                            tensor_x_train[0].unsqueeze(1),(
                            tensor_x_train[0].unsqueeze(1)).shape))

        """
        Choose MODEL and LOSS FUNCTION
        """
        if arch == "resnet":
            # model = CIFAR_ResNet34(num_class)
            # model = CIFAR_ResNet32(num_class)
            model = resnet32()
            learning_rate = 1e-1
        elif arch == "inception_small":
            model = CIFAR_Inception_Small(num_classes=num_class, batch_norm=batch_norm)
            learning_rate = 1e-1

        model = model.to(device)
        print(model)
    
        kwargs = {}

        if loss_name == "rll":
            kwargs['alpha'] = 0.1 # 0.45 # 0.01
        elif loss_name == "gce":
            kwargs['q'] = 0.7
        elif loss_name == "norm_mse":
            kwargs['alpha'] = 0.1
            kwargs['beta'] = 1.

        loss_fn = get_loss(loss_name, num_class, reduction="none", **kwargs)
        
        print("\n===========\nloss: {}\n===========\n".format(loss_name))

        if loss_name == "dmi":
            model.load_state_dict(torch.load(chkpt_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-cce-%s-nr-0%s-run-%s.pt" % (
                                        mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                                        noise_type, str(int(noise_rate * 10)), str(run)))['model_state_dict'])

        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr = learning_rate)


        lr_lambda = lambda epoch: (0.1**(epoch>=100)) * (0.1**(epoch>=150))
        lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #     factor=0.1, patience=5, verbose=True, threshold=0.0001,
        #     threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)


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

        """
        Aggregate sample-wise values for each batch
        """
        epoch_loss_train_agg = np.zeros((X_train.shape[0], num_epoch))
        epoch_acc_train_agg = np.zeros((X_train.shape[0], num_epoch))
        epoch_pred_train_agg = np.zeros((X_train.shape[0], num_epoch))

        for epoch in range(num_epoch):

            loss_train, acc_train, loss_train_agg, acc_train_agg, pred_train_agg = train(train_loader, model)

            # Logs
            writer.add_scalar('training_loss', loss_train, epoch)
            writer.add_scalar('training_accuracy', acc_train, epoch)
            writer.close()          

            # Validation
            loss_val, acc_val = test(val_loader, model, run, use_best=False)

            # Testing
            loss_test, acc_test = test(test_loader, model, run, use_best=False)

            # Logs
            writer.add_scalar('testing_loss', loss_test, epoch)
            writer.add_scalar('testing_accuracy', acc_test, epoch)
            writer.close()

            lr_scheduler.step()

            epoch_loss_train.append(loss_train)
            epoch_acc_train.append(acc_train)
            epoch_loss_train_agg[:, epoch] = np.asarray(loss_train_agg)
            epoch_acc_train_agg[:, epoch] = np.asarray(acc_train_agg)
            epoch_pred_train_agg[:, epoch] = np.asarray(pred_train_agg)

            epoch_loss_test.append(loss_test)
            epoch_acc_test.append(acc_test)


            if epoch == 0:
                best_acc_val = acc_val
            
            if best_acc_val < acc_val:

                best_model_wts = copy.deepcopy(model.state_dict())
                state_dict = {'model_state_dict': model.state_dict(), 
                                'opt_state_dict': optimizer.state_dict(),
                                'best_acc_val': best_acc_val,
                                'epoch': epoch,
                                'run': run
                            }
                torch.save(state_dict, chkpt_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                                        mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                                        loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
                print("Best model weights updated...\n")
                best_acc_val = acc_val

            print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                    optimizer.param_groups[0]['lr'], 
                                                    loss_train, loss_val, loss_test, 
                                                    acc_train, acc_val, acc_test))


        loss_test, acc_test = test(test_loader, model, run, use_best=False)

        print(f"Run: {run}::: Test set performance - test_acc: {acc_test}, test_loss: {loss_test}\n")

        if noise_rate > 0.:
            state_dict = {'model_state_dict': model.state_dict(), 
                        'opt_state_dict': optimizer.state_dict(),
                        'best_acc_val': best_acc_val,
                        'epoch': epoch,
                        'run': run
                        }
            torch.save(state_dict, chkpt_path + 
                        "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (
                        mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                        loss_name, noise_type, 
                        str(int(noise_rate * 10)), 
                        str(run)))


        """
        Save results
        """

        with open(res_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (
                            mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                            loss_name, noise_type, 
                            str(int(noise_rate * 10)), 
                            str(run)), 'wb') as f:
            if noise_rate > 0.:
                pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                            'epoch_acc_train': np.asarray(epoch_acc_train),
                            'epoch_loss_test': np.asarray(epoch_loss_test), 
                            'epoch_acc_test': np.asarray(epoch_acc_test), 
                            'epoch_loss_train_agg': epoch_loss_train_agg,
                            'epoch_acc_train_agg': epoch_acc_train_agg,
                            'epoch_pred_train_agg': epoch_pred_train_agg,
                            'idx_tr_clean_ref': idx_tr_clean_ref, 
                            'idx_tr_noisy_ref': idx_tr_noisy_ref,
                            'y_train_org': y_temp[idx_train], 
                            'y_train':y_train,
                            'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                            'epoch_acc_train': np.asarray(epoch_acc_train),
                            'epoch_loss_test': np.asarray(epoch_loss_test), 
                            'epoch_acc_test': np.asarray(epoch_acc_test), 
                            'epoch_loss_train_agg': epoch_loss_train_agg,
                            'epoch_acc_train_agg': epoch_acc_train_agg,
                            'epoch_pred_train_agg': epoch_pred_train_agg,
                            'y_train_org': y_temp[idx_train], 
                            'y_train':y_train,
                            'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Pickle file saved: " + res_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (
                    mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                    loss_name, noise_type, 
                    str(int(noise_rate * 10)), 
                    str(run)), "\n")

        # Print the elapsed time
        elapsed = time.time() - t_start
        print("\nelapsed time: \n", elapsed)

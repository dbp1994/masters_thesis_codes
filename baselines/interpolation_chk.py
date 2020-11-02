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

import numpy_indexed as npi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
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

# def lid(logits, k=20):
#     """
#     ::: Source: https://github.com/xingjunm/dimensionality-driven-learning/blob/master/callback_util.py :::

#     Calculate LID for a minibatch of training samples based on the outputs of the network.
    
#     :param logits:
#     :param k: 
#     :return: 
#     """
#     epsilon = 1e-12
#     batch_size = tf.shape(logits)[0]
#     # n_samples = logits.get_shape().as_list()
#     # calculate pairwise distance
#     r = tf.reduce_sum(logits * logits, 1)
#     # turn r into column vector
#     r1 = tf.reshape(r, [-1, 1])
#     D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
#         tf.ones([batch_size, batch_size])

#     # find the k nearest neighbor
#     D1 = -tf.sqrt(D)
#     D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
#     D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

#     m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
#     v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
#     lids = -k / v_log
#     return lids

# def mle_batch(data, batch, k):
#     """
#     ::: Source: https://github.com/xingjunm/dimensionality-driven-learning/blob/master/callback_util.py :::

#     lid of a batch of query points X.
#     numpy implementation.
#     :param data: 
#     :param batch: 
#     :param k: 
#     :return: 
#     """
#     data = np.asarray(data, dtype=np.float32)
#     batch = np.asarray(batch, dtype=np.float32)

#     k = min(k, len(data) - 1)
#     f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
#     a = scipy.spatial.distance.cdist(batch, data)
#     a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
#     a = np.apply_along_axis(f, axis=1, arr=a)
#     return a


# rand_idxes = np.random.choice(self.X_train.shape[0], self.lid_subset_size  = 1280, replace=False)    

# def get_lids_random_batch(model, X, k=20, batch_size=128):
#     """
#     ::: Source: https://github.com/xingjunm/dimensionality-driven-learning/blob/master/callback_util.py :::

#     Get the local intrinsic dimensionality of each Xi in X_adv
#     estimated by k close neighbours in the random batch it lies in.
#     :param model: if None: lid of raw inputs, otherwise LID of deep representations 
#     :param X: normal images 
#     :param k: the number of nearest neighbours for LID estimation  
#     :param batch_size: default 100
#     :return: lids: LID of normal images of shape (num_examples, lid_dim)
#             lids_adv: LID of advs images of shape (num_examples, lid_dim)
#     """
#     if model is None:
#         lids = []
#         n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#         for i_batch in range(n_batches):
#             start = i_batch * batch_size
#             end = np.minimum(len(X), (i_batch + 1) * batch_size)
#             X_batch = X[start:end].reshape((end - start, -1))

#             # Maximum likelihood estimation of local intrinsic dimensionality (LID)
#             lid_batch = mle_batch(X_batch, X_batch, k=k)
#             lids.extend(lid_batch)

#         lids = np.asarray(lids, dtype=np.float32)
#         return lids

#     # get deep representations
#     ## funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
#     ##          for out in [model.get_layer("lid").output]]
#     image_modules = list(model.children())[:-1]
#     modelA = nn.Sequential(*image_modules)
#     lid_dim = len(image_modules)

#     #     print("Number of layers to estimate: ", lid_dim)

#     def estimate(i_batch):
#         start = i_batch * batch_size
#         end = np.minimum(len(X), (i_batch + 1) * batch_size)
#         n_feed = end - start
#         lid_batch = np.zeros(shape=(n_feed, lid_dim))
#         for i, func in enumerate(funcs):
#             X_act = func([X[start:end], 0])[0]
#             X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))

#             # Maximum likelihood estimation of local intrinsic dimensionality (LID)
#             lid_batch[:, i] = mle_batch(X_act, X_act, k=k)

#         return lid_batch

#     lids = []
#     n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#     for i_batch in range(n_batches):
#         lid_batch = estimate(i_batch)
#         lids.extend(lid_batch)

#     lids = np.asarray(lids, dtype=np.float32)

#     return lids


# Model
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


def train(train_loader, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0
    # correct_clean = 0
    # correct_noisy = 0

    # loss_clean = 0.
    # acc_clean = 0.

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))

    model.train()

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)       
        # Transfer data to the GPU
        x, y = x.to(device), y.to(device)

        # print(x.shape, y.shape)
        # print(torch.squeeze(x,1).shape)
        # input("Press <ENTER>.\n")

        # y_true = F.one_hot(y.type(torch.LongTensor)).to(device)

        output = model(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_batch = loss_fn(output, y)
        # loss_batch = loss_fn(output, y_true)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        # loss_batch.backward() # - for L_DMI
        optimizer.step()

        # with torch.no_grad():
        #     optimizer.old_loss = loss_batch.mean().item()
        #     optimizer.curr_loss = loss_fn(model(x), y_true).mean().item()

        loss_train += torch.mean(loss_batch).item()
        # loss_train += loss_batch.item() # - for L_DMI
        correct += (pred.eq(y.to(device))).sum().item()

        loss_train_agg[list(map(int, idx.tolist()))] = np.asarray(loss_batch.tolist())
        acc_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.eq(y.to(device)).tolist())

        batch_cnt = batch_id + 1

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return loss_train, acc_train, loss_train_agg, acc_train_agg


def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    print("Testing...\n")

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt"
                            % (mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)
        
            x, y = x.to(device), y.to(device)

            # y_true = F.one_hot(y.type(torch.LongTensor)).to(device)

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



if __name__ == "__main__":

    T_const = 1000 # 10
    N_const = 50 # 10
    # 3e-3 for RLL and 2e-4 for CCE
    # 1e-2 for SGD
    learning_rate = 1e-2 # 2e-4 # 3e-3 # 3e-4 # 1e-2
    num_epoch = 200
    batch_size = 128 # 64
    num_runs = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    parser = argparse.ArgumentParser(description='PyTorch "Memorization Check in NNs" Training')
    parser.add_argument('-nr','--noise_rate', default=0.4, type=float, help='noise_rate')
    parser.add_argument('-nt','--noise_type', default="sym", type=str, help='noise_type')
    parser.add_argument('-loss','--loss_name', default="cce", type=str, help='loss_name')
    args = parser.parse_args()


    noise_rate = args.noise_rate # 0.6
    noise_type = args.noise_type # "sym"

    dataset = "mnist" # "checker_board" # "board" #  "mnist" # "board"
    mode = "interpolation"

    for run in range(num_runs):

        epoch_loss_train = []
        epoch_acc_train = []

        epoch_loss_test = []
        epoch_acc_test = []

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

            X_temp, y_temp = X, y
        else:
            raise SystemExit("Dataset not supported.\n")

        """
        Add LABEL NOISE
        """

        if noise_rate > 0.:
                if noise_type=="sym":
                    y_temp_noisy, _ = noisify_with_P(y_temp, num_class, noise_rate, random_state=42)
                else:
                    raise SystemExit("Noise type not supported.\n")
        else:
            y_temp_noisy = y_temp

        X_train, X_val, y_train, y_val = model_selection.train_test_split(X_temp, y_temp_noisy,
                            test_size=0.2, random_state=42)

        """
        Since the data has been shuffled during 'sklearn.model_selection',
        we wil keep track of the indices so as to infer clean and 
        noisily-labelled samples
        """

        if noise_rate > 0.:
            dat_noisy = np.concatenate((X_train, X_val), axis=0)
            print(dat_noisy.shape)
            if dataset in ["board", "checker_board"]:
                X = X_temp
            if dataset == "mnist":
                X = X_temp.reshape((X_temp.shape[0],784))
                dat_noisy = dat_noisy.reshape((dat_noisy.shape[0], 784))
            elif dataset in ["cifar10", "cifar100"]:
                X = X_temp.reshape((X_temp.shape[0],1024*3))
                dat_noisy = dat_noisy.reshape((dat_noisy.shape[0],1024*3))

            print(dat_noisy.shape)
            print(X.shape)

            idx = []
            idx_train_clean = []
            idx_train_noisy = []
            st = time.perf_counter()
            idx = npi.indices(X, dat_noisy)
            stp = time.perf_counter()
            tm = stp - st
            print("search time: ", tm,"\n")

            print("idx[5]: ", idx[5])
            print((X[idx[5]]==dat_noisy[5]).all())

            idx_train = idx[0:X_train.shape[0]]
            idx_val = idx[X_train.shape[0]:]
            print(len(idx_train))
            print(len(idx_val))
            print("y_train: {}".format(y_train.shape))


            idx_tr_clean_ref = np.where(y_temp[idx_train] == y_temp_noisy[idx_train])[0]
            idx_tr_noisy_ref = np.where(y_temp[idx_train] != y_temp_noisy[idx_train])[0]

            if dataset == "mnist":
                idx_train_clean = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_clean_ref,:])
                idx_train_noisy = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_noisy_ref,:])
            elif dataset in ["cifar10", "cifar100"]:
                idx_train_clean = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_clean_ref,:])
                idx_train_noisy = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_noisy_ref,:])
            
        else:
            idx = []
            idx_train = []
            idx_val = []
            idx_train_clean = []
            idx_train_noisy = []

        
        # for i in range(num_class):
        #     print("train - class %d : " %(i), (y_train[y_train==i]).shape, "\n")
        #     print("val - class %d : " %(i), (y_val[y_val==i]).shape, "\n")
        #     print("test - class %d : " %(i), (y_test[y_test==i]).shape, "\n")
        # input("Press <ENTER> to continue...\n")

        
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
        # tensor_id_train = torch.Tensor(idx_train)
        tensor_id_train = torch.from_numpy(np.asarray(list(range(X_train.shape[0]))))

        dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train, tensor_id_train)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # Val. set
        tensor_x_val = torch.Tensor(X_val)
        tensor_y_val = torch.Tensor(y_val)
        # tensor_id_val = torch.Tensor(idx_val)
        
        val_size = 100
        dataset_val = torch.utils.data.TensorDataset(tensor_x_val, tensor_y_val) #, tensor_id_val)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=val_size, shuffle=True)

        # Test set
        tensor_x_test = torch.Tensor(X_test)
        tensor_y_test = torch.Tensor(y_test)
        test_size = 100
        dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_size, shuffle=True)


        print("\n tensor_x_train[0].unsqueeze(1): {}, {}\n".format(
                            tensor_x_train[0].unsqueeze(1),(
                            tensor_x_train[0].unsqueeze(1)).shape))

        """
        Choose MODEL and LOSS FUNCTION
        """
        
        if dataset in ["board", "checker_board"]:
            model = mem_NN()
        elif dataset == "mnist":
            # for i in range(10):
            # model = MNIST_CNN()
            model = MNIST_ResNet18()
        elif dataset == "cifar10":
            model = torchvision.models.resnet18()
            learning_rate = 5e-2

        model = model.to(device)
        print(model)
        
        loss_name = args.loss_name # "cce" # "mae" # "dmi" # "cce" # "rll"
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

        # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
        # optimizer = Alopex(model.parameters(), lr = learning_rate, T_const=T_const, 
        #                             N_const=N_const, model=model)
        # opt = Alopex_2T(model.parameters(), lr = learning_rate, T_const=T_const, N_const=N_const, model=model)

        lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=0.1, patience=5, verbose=True, threshold=0.0001,
                    threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
        if dataset == "mnist":
            lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        elif dataset == "cifar10":
            lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
        lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


        _, acc = test(test_loader, model, run)
        print(f"Raw Test Acc.: {acc}")

        input("Press <ENTER> to continue.\n")

        
        # input("\nPress <ENTER> to start the training...\n")

        """
        Setting up Tensorbard
        """
        writer = SummaryWriter(log_dirs_path)
        writer.add_graph(model, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
        writer.close()


        best_acc_val = 0.

        """
        Aggregate sample-wise loss values for each epoch
        """
        epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
        epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))

        for epoch in range(num_epoch):

            loss_train, acc_train, loss_train_agg, acc_train_agg = train(train_loader, model)
            writer.add_scalar('training_loss', loss_train, epoch)
            writer.add_scalar('training_accuracy', acc_train, epoch)
            writer.close()

            # Validation
            loss_val, acc_val = test(val_loader, model, run, use_best=False)

            # Testing
            loss_test, acc_test = test(test_loader, model, run, use_best=False)
            writer.add_scalar('testing_loss', loss_test, epoch)
            writer.add_scalar('testing_accuracy', acc_test, epoch)
            writer.close()

            epoch_loss_train.append(loss_train)
            epoch_acc_train.append(acc_train)
            epoch_loss_train_agg[:, epoch] = loss_train_agg
            epoch_acc_train_agg[:, epoch] = acc_train_agg

            epoch_loss_test.append(loss_test)
            epoch_acc_test.append(acc_test)

            # Learning Rate Scheduler Update
            
            # lr_scheduler_1.step(loss_val)
            ## lr_scheduler_3.step()

            if dataset == "cifar10":
                lr_scheduler_2.step()


            if epoch == 0:
                best_acc_val = acc_val
            
            if best_acc_val < acc_val:

                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (
                                mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
                print("Model weights updated...\n")
                best_acc_val = acc_val

            print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                optimizer.param_groups[0]['lr'], 
                                                loss_train, loss_val, loss_test, 
                                                acc_train, acc_val, acc_test))


        loss_test, acc_test = test(test_loader, model, run, use_best=False)

        print(f"Run: {run}::: Test set performance - test_acc: {acc_test}, test_loss: {loss_test}\n")

        if noise_rate > 0.:
            torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (
                                mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run)))

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

    # fig1 = plt.figure(1)
    # plt.plot(list(range(len(epoch_loss_train))), epoch_loss_train, 'r')
    # plt.plot(list(range(len(epoch_loss_test))), epoch_loss_train, 'r')
    # plt.title("mem. chk - nr: {:.1f}, nt: {}".format(noise_rate, noise_type))
    # fig1.savefig("mem_nn_sym_loss_chk.png", format="png", dpi=600)
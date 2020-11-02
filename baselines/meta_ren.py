from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import pickle
from tqdm import tqdm
from collections import OrderedDict
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from numpy.random import RandomState

from data import *
from meta_layers import *
from adversarial_attacks import *
from losses import *


# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
torch.cuda.manual_seed_all(3459)
# tf.set_random_seed(3459)

eps = 1e-8


def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc


def meta_ren_train(train_loader, X_val, y_val, model, first_order=False):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    # meta_lr = 3e-4
    sample_wts = np.zeros((len(train_loader.dataset), ))

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))

    model.train()

    meta_net = MNIST_MetaNN()
    meta_net = meta_net.to(device)

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        y_val = y_val.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)
        x_val, y_val = X_val.to(device), y_val.to(device)

        # Load the current n/w params. into meta_net
        meta_net.load_state_dict(model.state_dict())

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_net(x)
        loss_pass_1 = loss_fn(y_f_hat, y)
        eps = torch.zeros(loss_pass_1.size(), requires_grad = True)
        eps = eps.to(device)
        l_f_meta = torch.sum(loss_pass_1 * eps)

        meta_net.zero_grad()

        # Line 6 - 7 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, meta_net.params(), create_graph=True)

        meta_net.update_params(meta_lr, source_params=grads)


#         (https://discuss.pytorch.org/t/replace-the-input-after-forward-but-before-backward/89199)

#         For the nn.Module version, note that the optimizer step and model state dict loading are both done in a non-differentiable manner. So these ops will be ignored by the autograd.
# When you call backward, you will get the gradient wrt to the Parameter that was used in the forward.

#         (https://discuss.pytorch.org/t/autograd-doesnt-retain-the-computation-graph-on-a-meta-learning-algorithm/77843/3)
#         Yes,
#         But the tricky bit is that nn.Parameter() are built to be parameters that you learn. So they cannot have history. 
#         So you will have to delete these and replace them with the new updated values as Tensors (and keep them in a 
#         different place so that you can still update them with your optimizer).
#         That is why I recommended the library above that does all that for you.


#         (https://discuss.pytorch.org/t/cannot-calculate-second-order-gradients-even-though-create-graph-true/78711)
#         I think the problem in your code is:

#         for p in m.parameters():
#             p = p - LR * p.grad
        
#     You can check the 'higher' package to do this properly (they solve this problem of nn.Parameter for you). 
#     Otherwise, if you want to do it manually, you can find some info in this thread: How does one have 
#     the parameters of a model NOT BE LEAFS? (https://discuss.pytorch.org/t/how-does-one-have-the-parameters-of-a-model-not-be-leafs/70076/9)


        # https://discuss.pytorch.org/t/layer-weight-vs-weight-data/24271

        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723

        # https://stackoverflow.com/questions/60311609/how-does-one-implement-a-meta-trainable-step-size-in-pytorch

        """
        Potential solution:

        https://github.com/cnguyen10/few_shot_meta_learning/blob/632e42c8cefb1b98222c8ae7e8e0086e0e524221/maml.py#L417

        https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/meta_neural_network_architectures.py#L208

        """

        # https://github.com/tristandeleu/pytorch-meta/tree/master/examples

        # https://github.com/MichaelMMeskhi/MtL-Progress-github.io

        # https://github.com/byu-dml/metalearn

        # https://github.com/tristandeleu/pytorch-maml

        # https://github.com/amzn/metalearn-leap/blob/master/src/maml/maml/maml.py

        # https://github.com/dragen1860/awesome-meta-learning

        # https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L43

        # https://github.com/learnables/learn2learn/blob/6d113f84386f13afc548ba2cd99a2a0e02935a23/learn2learn/utils.py#L205

        # https://discuss.pytorch.org/t/assign-variable-to-model-parameter/6150
        
        # https://github.com/szagoruyko/diracnets/blob/master/diracconv.py#L15-L41

        # https://discuss.pytorch.org/t/is-setattr-something-we-need-when-creating-custom-nn-with-changing-layers/14555

        # https://github.com/facebookresearch/higher

        # https://www.google.com/search?rlz=1C1SQJL_enIN887IN887&sxsrf=ALeKk001fTiJK5RuTOo16m8ORb5mGuv1RA%3A1595526448872&ei=MM0ZX9CbMLy-3LUPrpuMuA4&q=setattr+parameter+update+pytorch+&oq=setattr+parameter+update+pytorch+&gs_lcp=CgZwc3ktYWIQAzoECAAQRzoGCAAQDRAeOgYIABAWEB46CAgAEAgQDRAeUOUsWJFlYOhnaAFwAXgAgAHSAYgB2BuSAQYwLjE3LjKYAQCgAQGqAQdnd3Mtd2l6wAEB&sclient=psy-ab&ved=0ahUKEwjQ89aZ9-PqAhU8H7cAHa4NA-cQ4dUDCAw&uact=5

        # https://github.com/szagoruyko/functional-zoo/blob/master/resnet-18-export.ipynb


        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(x_val)
        # l_g_meta = torch.mean(cce_loss(y_g_hat, y_val))
        l_g_meta = torch.mean(loss_fn(y_g_hat, y_val))
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(torch.neg(grad_eps), min=0)
        norm_const = torch.sum(w_tilde) 

        if norm_const != 0:
            if norm_const < 1e-8:
                ex_wts = w_tilde / 1e-8
            else:
                ex_wts = w_tilde / norm_const
        else:
            ex_wts = w_tilde

        sample_wts[list(map(int, (idx.to('cpu')).tolist()))] = np.asarray((ex_wts.to('cpu')).tolist())

        # print("\n norm_const: {}\n".format(norm_const))
        # print("\n ex_wts: {}\n".format(ex_wts))
        # print("\n====================\n")
        # input("Press <ENTER> to continue.\n")

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update

        y_f_hat = model(x)
        loss_batch = loss_fn(y_f_hat, y)
        loss_meta_ren = torch.sum(loss_batch * ex_wts)

        optimizer.zero_grad()
        loss_meta_ren.backward()
        optimizer.step()

        # Compute the accuracy and loss after meta-updation
        pred_prob = F.softmax(y_f_hat, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_train += (loss_meta_ren).item() # .item() for scalars, .tolist() in general
        correct += pred.eq(y.to(device)).sum().item()

        loss_train_agg[list(map(int, idx.tolist()))] = np.asarray(loss_batch.tolist())
        acc_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.eq(y.to(device)).tolist())

        batch_cnt = batch_id + 1

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return sample_wts, loss_train, acc_train, loss_train_agg, acc_train_agg

def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (
                                mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run))))
                model = model.to(device)

            """
            Loss Function expects the labels to be 
            integers and not floats
            """
            y = y.type(torch.LongTensor)
        
            x, y = x.to(device), y.to(device)

            # x = x.reshape((-1, 784))

            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            batch_loss = loss_fn(output, y)
            loss_test += torch.mean(batch_loss).item()
            correct += (pred.eq(y.to(device))).sum().item()

            batch_cnt = batch_id + 1
        
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)

    return loss_test, acc_test


"""
Configuration
"""


parser = argparse.ArgumentParser(description = 'Meta Reweight- Ren et al. (ICML 18)')
parser.add_argument('-nr','--noise_rate', default=0.4, type=float, help='noise_rate')
parser.add_argument('-nt','--noise_type', default="sym", type=str, help='noise_type')
parser.add_argument('-loss','--loss_name', default="cce", type=str, help='loss_name')
args = parser.parse_args()


noise_rate = args.noise_rate # 0.6
noise_type = args.noise_type # "sym"
loss_name = args.loss_name 

dataset = "mnist" # "checker_board" # "board" #  "mnist" # "board"
mode = "meta_ren"

num_epoch = 150
num_runs = 1 # 5
batch_size = 128
# num_batches = int(X_train.shape[0] / batch_size)
learning_rate = 2e-4
random_state = 422
meta_lr = 2e-3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for run in range(num_runs):

    print("\n==================\n")
    print(f"== RUN No.: {run} ==")
    print("\n==================\n")

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []
    sample_wts_fin = []

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

    # Train. set
    tensor_x_train = torch.Tensor(X_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_y_train = torch.Tensor(y_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
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

    
    """
    Choose MODEL and LOSS FUNCTION
    """

    if dataset == "mnist":
        model = MNIST_MetaNN()
        # model = MNIST_NN()
    elif dataset == "cifar10":
        # model = CIFAR10_NN()
        pass
    # params = list(model.parameters())
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


    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    writer.add_graph(model, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
    writer.close()

    """
    Aggregate sample-wise loss values for each epoch
    """
    epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))

    for epoch in range(1, num_epoch+1):

        #Training set performance
        sample_wts, loss_train, acc_train, loss_train_agg, acc_train_agg = meta_ren_train(train_loader,
                                            tensor_x_val, tensor_y_val, loss_fn, model)

        ## log TRAIN. SET performance
        writer.add_scalar('training_loss', loss_train, epoch)
        writer.add_scalar('training_accuracy', acc_train, epoch)
        writer.close()

        # Validation set performance
        loss_val, acc_val = test(val_loader, loss_fn, model, use_best=False)

        #Testing set performance
        loss_test, acc_test = test(test_loader, loss_fn, model, use_best=False)

        ## log TEST SET performance
        writer.add_scalar('test_loss', loss_test, epoch)
        writer.add_scalar('test_accuracy', acc_test, epoch)
        writer.close()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)    

        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

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

        # Update best_acc_val and sample_wts_fin
        if epoch == 1:
            best_acc_val = acc_val
            sample_wts_fin = sample_wts
            # print(sample_wts)
        else:
            sample_wts_fin = np.concatenate((sample_wts_fin, sample_wts), axis=0)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (
                                mode, dataset, loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
            print("Model weights updated...\n")

        print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch, 
                                                optimizer.param_groups[0]['lr'], 
                                                loss_train, loss_val, loss_test, 
                                                acc_train, acc_val, acc_test))


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
            pickle.dump({'sample_wts_fin': sample_wts_fin,
                        'epoch_loss_train': np.asarray(epoch_loss_train), 
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
            # pickle.dump({'sample_wts_fin': sample_wts_fin,
            #             'epoch_loss_train': np.asarray(epoch_loss_train), 
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
            pickle.dump({'sample_wts_fin': sample_wts_fin,
                        'epoch_loss_train': np.asarray(epoch_loss_train), 
                        'epoch_acc_train': np.asarray(epoch_acc_train),
                        'epoch_loss_test': np.asarray(epoch_loss_test), 
                        'epoch_acc_test': np.asarray(epoch_acc_test), 
                        'epoch_loss_train_agg': epoch_loss_train_agg,
                        'epoch_acc_train_agg': epoch_acc_train_agg,
                        'X_temp': X_temp, 'X_train': X_train, 'X_val': X_val,
                        'y_temp': y_temp, 'y_train':y_train, 'y_val':y_val,
                        'num_epoch': num_epoch}, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump({'sample_wts_fin': sample_wts_fin,
            #             'epoch_loss_train': np.asarray(epoch_loss_train), 
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


"""
You can compute the F-score yourself in pytorch. The F1-score is defined for single-class (true/false) 
classification only. The only thing you need is to aggregate the number of:
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
f1_score = 2 * precision * recall / (precision + recall)

"""
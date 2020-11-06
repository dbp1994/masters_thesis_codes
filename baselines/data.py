import numpy as np
from numpy.random import RandomState

from sklearn import model_selection

from add_noise import noisify_with_P, multiclass_noisify, noisify_mnist_asymmetric, noisify_cifar10_asymmetric

import torch
import torchvision
import torchtext
from torchvision import transforms

import numpy_indexed as npi
import time

# set seed for reproducibility
np.random.seed(3459)

def numpy_to_categorical(y, num_classes=None, dtype='float32'):
    """ 
    Taken from Keras repo
    https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9-L37
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def read_data(noise_type, noise_rate, dataset, data_aug=False, mode="rm"):

    if dataset == "mnist":

        num_class = 10
        
        if not data_aug:
            dat_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307, ), (0.3081, ))]))

            dat_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307, ), (0.3081, ))]))
            
            print("\nNO DATA AUGMENTATION...\n")
        
        else:
            raise NotImplementedError("Data augmentation not implemented.\n")
        
        X_temp = (dat_train.data).numpy()
        y_temp = (dat_train.targets).numpy()

        X_test = (dat_test.data).numpy()
        y_test = (dat_test.targets).numpy()

    elif dataset == "cifar10":
        num_class = 10

        if not data_aug:
            dat_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2023, 0.1994, 0.2010))]))

            dat_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2023, 0.1994, 0.2010))]))

            print("\nDATA AUGMENTATION DISABLED...\n")

        else:

            dat_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=
                                    transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2023, 0.1994, 0.2010))]))

            dat_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2023, 0.1994, 0.2010))]))

            print("\nDATA AUGMENTATION ENABLED...\n")
        
        X_temp = dat_train.data
        y_temp = np.asarray(dat_train.targets)

        X_test = dat_test.data
        y_test = np.asarray(dat_test.targets)

    elif dataset in ["board", "checker_board"]:
            dat = np.loadtxt(f"./{dataset}/{dataset}_data.txt", delimiter=",")
            X, y = dat[:,:-1], dat[:,-1]

            if int(np.min(y)) == 0:
                num_class = int(np.max(y) + 1)
            else:
                num_class = int(np.max(y))
            
            X_temp, X_test, y_temp, y_test = model_selection.train_test_split(X, y, 
                                test_size=0.2, random_state=42)
    
        # elif dataset == "imdb":
        #     dat_train = torchtext.datasets.IMDB('./data/', train=True, download=True)

        # elif dataset == "agnews":
        #     dat_train, dat_test = torchtext.datasets.AG_News(root='./data/', ngram=3, vocab=False)

    else:
        raise SystemExit("Dataset not supported.\n")


    #num_class = np.max(y_temp) + 1

    """
    Conv layers in PyTorch expect the data to be 4-D wherein #channels
    is the additional dimension
    i.e. (n_samples, channels, height, width)
    """
    if dataset == "mnist":
        X_temp = np.expand_dims(X_temp, 1)  # if numpy array
        # X_temp = X_temp.unsqueeze(1)  # if torch tensor

        X_test = np.expand_dims(X_test, 1)  # if numpy array
        # X_test = X_test.unsqueeze(1)  # if torch tensor

    elif dataset in ["cifar10", "cifar100"]:
        X_temp = X_temp.transpose(0, 3, 1, 2)   # if numpy array
        # X_temp = X_temp.permute(0, 3, 1, 2)   # if torch tensor

        X_test = X_test.transpose(0, 3, 1, 2)   # if numpy array
        # X_test = X_test.permute(0, 3, 1, 2)   # if torch tensor
    

    """
    Add Label Noise
    """

    if mode in ["rm", "active_bias", "batch_rewgt", "selfie", "meta_mlnt", "pencil"]:

        if noise_rate > 0.:
            if noise_type=='sym':
                y_temp_noisy, P = noisify_with_P(y_temp, num_class, noise_rate, random_state=42)
            elif noise_type == 'cc':
                if dataset == "mnist":
                    y_temp_noisy, P = noisify_mnist_asymmetric(y_temp, noise_rate, random_state=42)
                elif dataset == "cifar10":
                    y_temp_noisy, P = noisify_cifar10_asymmetric(y_temp, noise_rate, random_state=42)
            else:
                raise SystemExit("Noise type not supported.")

        else:
            y_temp_noisy = y_temp
        
        X_train, X_val, y_train, y_val = model_selection.train_test_split(
                            X_temp, y_temp_noisy, test_size=0.2, random_state=42)


        ##y_train_orig = y_temp.copy()

        # y_train = numpy_to_categorical(y_train, num_class)
        ## The following works only for torch.Tensors
        ## y_train = torch.nn.functional.one_hot(torch.Tensor(y_train), num_class)

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
            elif dataset == "mnist":
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

            if dataset in ["board", "checker_board"]:
                idx_train_clean = npi.indices(X, X_train.reshape(-1, 2)[idx_tr_clean_ref,:])
                idx_train_noisy = npi.indices(X, X_train.reshape(-1, 2)[idx_tr_noisy_ref,:])
            if dataset == "mnist":
                idx_train_clean = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_clean_ref,:])
                idx_train_noisy = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_noisy_ref,:])
            elif dataset in ["cifar10", "cifar100"]:
                idx_train_clean = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_clean_ref,:])
                idx_train_noisy = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_noisy_ref,:])

            
            ids = (idx, idx_train, idx_val, idx_tr_clean_ref, idx_tr_noisy_ref, idx_train_clean, idx_train_noisy, P)
            
        else:
            idx = np.asarray(list(range(X_temp.shape[0])))
            idx_train = np.asarray(list(range(X_train.shape[0])))
            idx_val = np.asarray(list(range(X_val.shape[0])))
            idx_train_clean = []
            idx_train_noisy = []

            ids = (idx, idx_train, idx_val, idx_train_clean, idx_train_noisy)
        


        dat = (X_temp, y_temp, X_train, y_train, X_val, y_val, X_test, y_test)

    elif mode in ["meta_ren", "meta_net"]:
        num_val_clean = 1000
        num_hyp_val = 5000
        idx_val_clean = []

        y_temp = y_temp.astype(np.int32)

        # Pick equal no. clean samples from each class for val. set
        if dataset in ["mnist", "cifar10", "cifar100"]:
            num_class = 10
            for i in range(num_class):
                idx_cls_tmp = np.where(y_temp == i)[0]
                rng = np.random.default_rng()
                tmp_pick = rng.choice(idx_cls_tmp.shape[0], size=num_val_clean//num_class, replace=False)
                # tmp_pick = np.random.randint(low=0, high=idx_cls_tmp.shape[0], size=(num_val_clean//num_class,))
                if i == 0:
                  idx_val_clean = idx_cls_tmp[tmp_pick]
                else:
                  idx_val_clean = np.concatenate((idx_val_clean, idx_cls_tmp[tmp_pick]), axis=0)

        # Train. and Val. data are from the same set, so separate them out
        X_val = (X_temp[idx_val_clean,:,:,:]).copy()
        y_val = (y_temp[idx_val_clean]).copy()
        # X_train = X_temp.copy()
        # y_train = y_temp.copy()
        X_train_tmp = np.delete(X_temp, idx_val_clean, axis=0)
        y_train_tmp = np.delete(y_temp, idx_val_clean, axis=0)

        # Remove the extra 9000 samples as well for a fair comparison with
        # the other algorithms
        X_train, _, y_train, _ = model_selection.train_test_split(X_train_tmp, 
                                y_train_tmp, test_size=0.15253, random_state=42)

        print("\nX_temp: {}\n".format(X_temp.shape))
        print("y_temp: {}\n".format(y_temp.shape))
        print("X_train: {}\n".format(X_train.shape))
        print("y_train: {}\n".format(y_train.shape))
        print("X_val: {}\n".format(X_val.shape))
        print("y_val: {}\n".format(y_val.shape))

        # Training set indices
        idx_train = []
        st = time.perf_counter()
        if dataset == "mnist":
            idx_train = npi.indices(X_train_tmp.reshape((-1, 784)), X_train.reshape((-1, 784)))
            idx_train_ref = npi.indices(X_train.reshape((-1, 784)), (X_train_tmp.reshape((-1, 784)))[idx_train,:])
        elif dataset in ["cifar10", "cifar100"]:
            idx_train = npi.indices(X_train_tmp.reshape((-1, 1024*3)), X_train.reshape((-1, 1024*3)))
            idx_train_ref = npi.indices(X_train.reshape((-1, 1024*3)), (X_train_tmp.reshape((-1, 1024*3)))[idx_train,:])

        stp = time.perf_counter()
        tm = stp - st
        print("search time: ", tm,"\n")

        print("idx[5]: ", idx_train[5])
        print((X_train_tmp[idx_train[5]].reshape(-1, 784)==X_train[5].reshape(-1, 784)).all())


        if noise_rate > 0.:
            if noise_type=='sym':
                y_train_noisy, P = noisify_with_P(y_train, num_class, noise_rate, random_state=42)
            elif noise_type == 'cc':
                if dataset == "mnist":
                    y_train_noisy, P = noisify_mnist_asymmetric(y_train, noise_rate, random_state=42)
                elif dataset == "cifar10":
                    y_train_noisy, P = noisify_cifar10_asymmetric(y_train, noise_rate, random_state=42)
            else:
                raise SystemExit("Noise type not supported.")
            

            idx_train_clean_ref = np.where(y_train_tmp[idx_train]==y_train_noisy)[0]
            idx_train_noisy_ref = np.asarray(list(set(range(y_train_tmp[idx_train].shape[0])).difference(set(idx_train_clean_ref))))

            if dataset == "mnist":
                idx_train_clean = npi.indices(X_train_tmp.reshape((-1, 784)), 
                                (X_train.reshape((-1, 784)))[idx_train_clean_ref, :])
                idx_train_noisy = npi.indices(X_train_tmp.reshape((-1, 784)), 
                                (X_train.reshape((-1, 784)))[idx_train_noisy_ref, :])
            elif dataset in ["cifar10", "cifar100"]:
                idx_train_clean = npi.indices(X_train_tmp.reshape((-1, 1024*3)), 
                                (X_train.reshape((-1, 1024*3)))[idx_train_clean_ref, :])
                idx_train_noisy = npi.indices(X_train_tmp.reshape((-1, 1024*3)), 
                                (X_train.reshape((-1, 1024*3)))[idx_train_noisy_ref, :])

        else:
            y_train_noisy = y_train

            idx_train_clean_ref = np.asarray(list(range(y_train.shape[0]))).astype(np.int32)
            idx_train_noisy_ref = []

            idx_train_clean = idx_train
            idx_train_noisy = []

        """
        =====================================================================
        How are we choosing the hyper-validation set? Randomly from the training set?
        =====================================================================
        """
        # Hyper-validation set
        X_hyp_val = (X_train[5000,:,:,:]).copy()
        y_hyp_val = (y_train[5000]).copy()

        dat = (X_temp, y_temp, X_train, y_train_noisy, X_val, 
                    y_val, X_test, y_test, X_hyp_val, y_hyp_val)
        ids = (idx_train, idx_train_ref, idx_train_clean, idx_train_clean_ref, 
            idx_train_noisy, idx_train_noisy_ref, idx_val_clean)
    else:
        raise SystemExit("Mode not supported. Choose from: ['rm', \
                         'active_bias, 'batch_rewgt', 'meta_ren', \
                        'meta_mlnt', 'meta_net', \
                        'selfie', 'pencil']\n")
    
    # y_val = numpy_to_categorical(y_val, num_class)
    # y_test = numpy_to_categorical(y_test, num_class)

    ## The following works only for torch.Tensors
    ## y_val = torch.nn.functional.one_hot(torch.Tensor(y_val), num_class)
    ## y_test = torch.nn.functional.one_hot(torch.Tensor(y_test), num_class)



    return dat, ids
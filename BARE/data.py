import numpy as np
from numpy.random import RandomState
from sklearn import model_selection
import scipy.io as sio
import os
import os.path
import pickle
import time
from subprocess import call

from add_noise import noisify_with_P, multiclass_noisify, noisify_mnist_asymmetric, noisify_mnist_asymmetric_sent, noisify_cifar10_asymmetric, noisify_cifar10_asymmetric_sent, noisify_news_asymmetric, get_instance_noisy_label, noisify_pairflip

import torch
import torchvision
import torchtext
from torchvision import transforms
import torch.utils.data as data2

from PIL import Image

import numpy_indexed as npi


# set seed for reproducibility
np.random.seed(3459)

def default_loader(path):
        return Image.open(path).convert('RGB')

def default_flist_reader(flist):
        """
        flist format: impath label\nimpath label\n ...(same to caffe's filelist)
        """
        imlist = []
        with open(flist, 'r') as rf:
                for line in rf.readlines():
                        impath, imlabel = line.strip().split()
                        imlist.append( (impath, int(imlabel)) )
                                        
        return imlist

class ImageFilelist(data2.Dataset):
        def __init__(self, root, flist, transform=None, target_transform=None,
                        flist_reader=default_flist_reader, loader=default_loader):
                self.root   = root
                self.imlist = flist_reader(flist)                
                self.transform = transform
                self.target_transform = target_transform
                self.loader = loader

        def __getitem__(self, index):
                impath, target = self.imlist[index]
                img = self.loader(os.path.join(self.root,impath))
                if self.transform is not None:
                        img = self.transform(img)
                if self.target_transform is not None:
                        target = self.target_transform(target)
                
                return img, target, index

        def __len__(self):
                return len(self.imlist)


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


def read_data(noise_type, noise_rate, dataset, data_aug=False, mode="risk_min"):

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
        
        feat_size = 28 * 28

    elif dataset == "cifar10":
        num_class = 10

        # data_aug = True

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

            # dat_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=
            #                         transforms.Compose([transforms.CenterCrop(28),
            #                         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            #                         transforms.Normalize((0.4914, 0.4822, 0.4465), 
            #                         (0.2023, 0.1994, 0.2010))]))

            dat_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2023, 0.1994, 0.2010))]))
            

            print("\nDATA AUGMENTATION ENABLED...\n")
        
        X_temp = dat_train.data
        y_temp = np.asarray(dat_train.targets)

        X_test = dat_test.data
        y_test = np.asarray(dat_test.targets)

        feat_size = 3 * 32 * 32
    
    elif dataset == "cifar100":

        num_class = 100

        # stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

        if not data_aug:
            dat_train = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]))

            dat_test = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]))

            print("\nDATA AUGMENTATION DISABLED...\n")

        else:
            dat_train = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=
                                        transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]))               

            # dat_train = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=
            #                         transforms.Compose([transforms.CenterCrop(28),
            #                         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            #                         transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]))

            dat_test = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=
                                    transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]))
                    

            print("\nDATA AUGMENTATION ENABLED...\n")
                
        X_temp = dat_train.data
        y_temp = np.asarray(dat_train.targets)

        X_test = dat_test.data
        y_test = np.asarray(dat_test.targets)

        feat_size = 3 * 32 * 32
    
    # elif dataset == "imagenet_tiny":
    #     data_root = './data/imagenet-tiny/tiny-imagenet-200'
    #     train_kv = "train_noisy_%s_%s_kv_list.txt" % (noise_type, noise_rate) 
    #     test_kv = "val_kv_list.txt"

    #     normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
    #                                     std =[0.2302, 0.2265, 0.2262])

    #     train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv),
    #             transform=transforms.Compose([transforms.RandomResizedCrop(56),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(), 
    #             normalize,
    #     ]))

    #     test_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, test_kv),
    #             transform=transforms.Compose([transforms.Resize(64),
    #             transforms.CenterCrop(56),
    #             transforms.ToTensor(),
    #             normalize,
    #     ]))


    
    
    # elif dataset == "imdb":
    #     dat_train = torchtext.datasets.IMDB('./data/', train=True, download=True)

    # elif dataset == "agnews":
    #     dat_train, dat_test = torchtext.datasets.AG_News(root='./data/', ngram=3, vocab=False)

    elif dataset == "news":

        num_class = 6

        def regroup_dataset(labels):
            """
            categories = dataset.target_names
            labels = [(dataset.target_names.index(cat), cat) for cat in categories]

            [(0, 'alt.atheism'), (1, 'comp.graphics'), (2, 'comp.os.ms-windows.misc'), 
            (3, 'comp.sys.ibm.pc.hardware'), (4, 'comp.sys.mac.hardware'), (5, 'comp.windows.x'), 
            (6, 'misc.forsale'), (7, 'rec.autos'), (8, 'rec.motorcycles'), (9, 'rec.sport.baseball'), 
            (10, 'rec.sport.hockey'), (11, 'sci.crypt'), (12, 'sci.electronics'), (13, 'sci.med'), 
            (14, 'sci.space'), (15, 'soc.religion.christian'), (16, 'talk.politics.guns'), 
            (17, 'talk.politics.mideast'), (18, 'talk.politics.misc'), (19, 'talk.religion.misc')]
            """
            batch_y = labels.copy()
            for i, label in enumerate(labels):
                if label in [0, 15, 19]:
                    batch_y[i]=0
                if label in [1, 2, 3, 4, 5,]:
                    batch_y[i]=1
                if label in [6]:
                    batch_y[i]=2
                if label in [7,8,9,10]:
                    batch_y[i]=3
                if label in [11,12,13,14]:
                    batch_y[i]=4
                if label in [16,17,18]:
                    batch_y[i]=5
                    
            print('regrouped label', batch_y.shape)
            return batch_y

        weights_matrix, X1, y1 = pickle.load(open("./data/news.pkl", 'rb'), encoding='iso-8859-1')
        y1 = regroup_dataset(y1)

        X_temp, X_test, y_temp, y_test = model_selection.train_test_split(X1, y1, 
                                test_size=0.2, random_state=42)

        feat_size = 28 * 28
    
    elif dataset == "svhn":

        # dat_train = torchvision.datasets.SVHN('.\data', split='train', 
        # transform=transforms.Compose([transforms.ToTensor(), 
        # transforms.Normalize((111.61, ), (113.16, ), (120.565, ))]), download=True)

        if not os.path.isfile("./data/svhn_train.mat"):
            print('Downloading SVHN train set...')
            call(
                "curl -o data/svhn_train.mat "
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                shell=True
            )
        if not os.path.isfile("./data/svhn_test.mat"):
            print('Downloading SVHN test set...')
            call(
                "curl -o data/svhn_test.mat "
                "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                shell=True
            )
        train = sio.loadmat('./data/svhn_train.mat')
        test = sio.loadmat('./data/svhn_test.mat')
        X_temp = np.transpose(train['X'], axes=[3, 2, 0, 1])
        X_test = np.transpose(test['X'], axes=[3, 2, 0, 1])

        X_temp = X_temp / 255.0
        X_test = X_test / 255.0

        means = X_temp.mean(axis=0)
        # std = np.std(X_train)
        X_temp = (X_temp - means)  # / std
        X_test = (X_test - means)  # / std

        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        y_temp = np.reshape(train['y'], (-1,)) - 1
        y_test = np.reshape(test['y'], (-1,)) - 1


        feat_size = 3 * 32 * 32

    elif dataset == "fmnist":
        pass
        # torchvision.datasets.FashionMNIST('./data/', train=True, transform=None, target_transform=None, download=False)

        # if not data_aug:
        #     dat_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=
        #                             transforms.Compose([transforms.ToTensor(), 
        #                             transforms.Normalize((0.1307, ), (0.3081, ))]))

        #     dat_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=
        #                             transforms.Compose([transforms.ToTensor(), 
        #                             transforms.Normalize((0.1307, ), (0.3081, ))]))
            
        #     print("\nNO DATA AUGMENTATION...\n")
        
        # else:
        #     raise NotImplementedError("Data augmentation not implemented.\n")
        
        # X_temp = (dat_train.data).numpy()
        # y_temp = (dat_train.targets).numpy()

        # X_test = (dat_test.data).numpy()
        # y_test = (dat_test.targets).numpy()

        # feat_size = 28 * 28

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

    # if mode in ["risk_min", "active_bias", "batch_rewgt", "selfie", "meta_mlnt", "pencil", "dividemix", "loss_sent", "loss_reg_pat"]:
    if mode in ["meta_ren", "meta_net"]:
        num_val_clean = 1000
        idx_val_clean = []

        y_temp = y_temp.astype(np.int32)

        # Pick equal no. clean samples from each class for val. set
        if dataset in ["mnist", "cifar10", "cifar100", "news", "svhn"]:

            if dataset == "news":
                X_train, X_val, y_train, y_val = model_selection.train_test_split(
                                X_temp, y_temp, test_size=0.3333, random_state=42) 
                
                # print("===========================\n")
                # print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
                # print("\n===========================\n")
                num_class = 6
            elif dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
                X_train, X_val, y_train, y_val = model_selection.train_test_split(
                                X_temp, y_temp, test_size=0.2, random_state=42)

                if dataset == "cifar100": 
                    num_class = 100
                else:
                    num_class = 10
            else:
                raise NotImplementedError("Train/Val/Test split rule not specified.\n")

            for i in range(num_class):
                idx_cls_tmp = np.where(y_val == i)[0]
                rng = np.random.default_rng()
                tmp_pick = rng.choice(idx_cls_tmp.shape[0], size=num_val_clean//num_class, replace=False)
                # tmp_pick = np.random.randint(low=0, high=idx_cls_tmp.shape[0], size=(num_val_clean//num_class,))
                if i == 0:
                  idx_val_clean = idx_cls_tmp[tmp_pick]
                else:
                  idx_val_clean = np.concatenate((idx_val_clean, idx_cls_tmp[tmp_pick]), axis=0)
        else:
            raise SystemExit("Dataset not supported.\n")

        # Train. and Val. data are from the same set, so separate them out
        X_val = (X_val[idx_val_clean,:,:,:]).copy()
        y_val = (y_val[idx_val_clean]).copy()
        

        if noise_rate > 0.:
            if noise_type=='sym':
                y_train_noisy, P = noisify_with_P(y_train, num_class, noise_rate, random_state=42)
            elif noise_type == 'cc':
                if dataset in ["mnist", "svhn"]:
                    y_train_noisy, P = noisify_mnist_asymmetric(y_train, noise_rate, random_state=42)
                elif dataset == "cifar10":
                    y_train_noisy, P = noisify_cifar10_asymmetric(y_train, noise_rate, random_state=42)
                elif dataset == "cifar100":
                    y_train_noisy, _ = noisify_pairflip(y_train, noise_rate, random_state=42, nb_classes=num_class)
                elif dataset == "news":
                    y_train_noisy, P = noisify_news_asymmetric(y_train, noise_rate, random_state=42)
                else:
                    raise NotImplementedError("Not implemented for this datasets.")
            elif noise_type == "idn":
                y_train_noisy = get_instance_noisy_label(noise_rate, zip(X_train, y_train), y_train, num_class, feat_size, norm_std=0.1)
            else:
                raise SystemExit("Noise type not supported.")
            
        else:
            y_train_noisy = y_train
    
    else:

        num_val_clean = 1000
        idx_val_clean = []

        y_temp = y_temp.astype(np.int32)

        # Pick equal no. clean samples from each class for val. set
        if dataset in ["mnist", "cifar10", "cifar100", "news", "svhn"]:

            if dataset == "news":
                X_train, X_val, y_train, y_val = model_selection.train_test_split(
                                X_temp, y_temp, test_size=0.3333, random_state=42) 

                num_class = 6
            elif dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
                X_train, X_val, y_train, y_val = model_selection.train_test_split(
                                X_temp, y_temp, test_size=0.2, random_state=42)

                if dataset == "cifar100":
                    num_class = 100
                else:
                    num_class = 10
            else:
                raise NotImplementedError("Train/Val/Test split rule not specified.\n")

            for i in range(num_class):
                idx_cls_tmp = np.where(y_val == i)[0]
                rng = np.random.default_rng()
                tmp_pick = rng.choice(idx_cls_tmp.shape[0], size=num_val_clean//num_class, replace=False)
                # tmp_pick = np.random.randint(low=0, high=idx_cls_tmp.shape[0], size=(num_val_clean//num_class,))
                if i == 0:
                  idx_val_clean = idx_cls_tmp[tmp_pick]
                else:
                  idx_val_clean = np.concatenate((idx_val_clean, idx_cls_tmp[tmp_pick]), axis=0)

        else:
            raise SystemExit("Dataset not supported.\n")

        # Train. and Val. data are from the same set, so separate them out
        if dataset == "news":
            X_val = (X_val[idx_val_clean,:]).copy()
        elif dataset in ["mnist", "cifar10", "cifar100", "svhn"]:
            X_val = (X_val[idx_val_clean,:,:,:]).copy()
        y_val = (y_val[idx_val_clean]).copy()

        y_val_shape = y_val.shape[0]

        # print("===========================\n")
        # print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
        # print("\n===========================\n")

        y_temp1 = np.concatenate((y_train, y_val))

        if noise_rate > 0.:
            if noise_type=='sym':
                y_temp_noisy, P = noisify_with_P(y_temp1, num_class, noise_rate, random_state=42)
            elif noise_type == 'cc':
                if dataset in ["mnist", "svhn"]:
                    y_temp_noisy, P = noisify_mnist_asymmetric(y_temp1, noise_rate, random_state=42)
                elif dataset == "cifar10":
                    y_temp_noisy, P = noisify_cifar10_asymmetric(y_temp1, noise_rate, random_state=42)
                elif dataset == "cifar100":
                    y_train_noisy, _ = noisify_pairflip(y_train, noise_rate, random_state=42, nb_classes=num_class)
                elif dataset == "news":
                    y_temp_noisy, P = noisify_news_asymmetric(y_temp1, noise_rate, random_state=42)
                else:
                    raise NotImplementedError("Not implemented for this datasets.")
            elif noise_type == "idn":
                X_t = np.concatenate((X_train, X_val), axis=0)
                y_temp_noisy = get_instance_noisy_label(noise_rate, zip(X_t, y_temp1), y_temp1, num_class, feat_size, norm_std=0.1)
            else:
                raise SystemExit("Noise type not supported.")

        else:
            y_temp_noisy = y_temp1

        
        y_train_noisy = y_temp_noisy[:-y_val_shape]
        y_val = y_temp_noisy[-y_val_shape:]
        
    """
    Since the data has been shuffled during 'sklearn.model_selection',
    we wil keep track of the indices so as to infer clean and 
    noisily-labelled samples
    """

    if noise_rate > 0.:
        dat_noisy = np.concatenate((X_train, X_val), axis=0)
        print(dat_noisy.shape)
        if dataset in ["board", "checker_board", "news"]:
            X = X_temp
        elif dataset == "mnist":
            X = X_temp.reshape((X_temp.shape[0],784))
            dat_noisy = dat_noisy.reshape((dat_noisy.shape[0], 784))
        elif dataset in ["cifar10", "cifar100", "svhn"]:
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
        print("y_train: {}".format(y_train_noisy.shape))


        idx_tr_clean_ref = np.where(y_temp[idx_train] == y_train_noisy)[0]
        idx_tr_noisy_ref = np.where(y_temp[idx_train] != y_train_noisy)[0]

        if dataset in ["board", "checker_board"]:
            idx_train_clean = npi.indices(X, X_train.reshape(-1, 2)[idx_tr_clean_ref,:])
            idx_train_noisy = npi.indices(X, X_train.reshape(-1, 2)[idx_tr_noisy_ref,:])
        if dataset == "mnist":
            idx_train_clean = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_clean_ref,:])
            idx_train_noisy = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_noisy_ref,:])
        elif dataset in ["cifar10", "cifar100", "svhn"]:
            idx_train_clean = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_clean_ref,:])
            idx_train_noisy = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_noisy_ref,:])
        elif dataset == "news":
            idx_train_clean = npi.indices(X, X_train.reshape(-1, 1000)[idx_tr_clean_ref,:])
            idx_train_noisy = npi.indices(X, X_train.reshape(-1, 1000)[idx_tr_noisy_ref,:])

        
        ids = (idx, idx_train, idx_val, idx_tr_clean_ref, idx_tr_noisy_ref, idx_train_clean, idx_train_noisy, P)
        
    else:
        idx = np.asarray(list(range(X_temp.shape[0])))
        idx_train = np.asarray(list(range(X_train.shape[0])))
        idx_val = np.asarray(list(range(X_val.shape[0])))
        idx_train_clean = []
        idx_train_noisy = []

        ids = (idx, idx_train, idx_val, idx_train_clean, idx_train_noisy)
    

    if dataset == "news":
        dat = (X_temp, y_temp, X_train, y_train_noisy, X_val, y_val, X_test, y_test, weights_matrix)
    else:
        dat = (X_temp, y_temp, X_train, y_train_noisy, X_val, y_val, X_test, y_test)

    
    return dat, ids

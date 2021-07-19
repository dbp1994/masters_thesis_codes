import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.random import RandomState
from scipy import stats
from math import inf

# set seed for reproducibility
np.random.seed(3459)


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def noisify_with_P(y_train, nb_classes, noise, random_state=None):

    if noise > 0.0:
        P = build_uniform_P(nb_classes, noise)
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[int(i), :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """

    """
    Source: https://github.com/xingruiyu/coteaching_plus/blob/master/data/utils.py
    """

    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, P #, actual_noise

def noisify_mnist_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P

def noisify_cifar10_asymmetric_sent(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    # n = noise

    n1 = 0.2
    n2 = 0.3
    n3 = 0.4

    # automobile <- truck
    P[9, 9], P[9, 1], P[9, 7] = 1. - n1, n1/2, n1/2

    # bird -> airplane
    P[2, 2], P[2, 0], P[2, 6] = 1. - n2, 0.2, 0.1

    # cat <-> dog
    P[3, 3], P[3, 5], P[3, 8], P[3, 0] = 1. - n3, 0.1, 0.2, 0.1
    P[5, 5], P[5, 3], P[5, 9] = 1. - n3, 0.1, 0.3

    # deer -> horse
    P[4, 4], P[4, 7], P[4, 1], P[4, 2] = 1. - n2, 0.1, 0.1, 0.1

    y_train_noisy = multiclass_noisify(y_train, P=P,
                                        random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy

    return y_train, P

def noisify_mnist_asymmetric_sent(y_train, noise, random_state=None):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    n1 = 0.5
    n2 = 0.4
    n3 = 0.45

    # 1 <- 7
    P[7, 7], P[7, 1] , P[7, 9] = 1. - n1, n1/2, n1/2

    # 2 -> 7
    P[2, 2], P[2, 7], P[2, 9] = 1. - n2, 0.3, 0.1

    # 5 <-> 6
    P[5, 5], P[5, 6], P[5, 4] = 1. - n3, 0.3, 0.15
    P[6, 6], P[6, 5], P[6, 4] = 1. - n3, 0.35, 0.1

    # 3 -> 8
    P[3, 3], P[3, 8], P[3, 5] = 1. - n1, 0.4, 0.1

    y_train_noisy = multiclass_noisify(y_train, P=P,
                                        random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy

    return y_train, P



def noisify_news_asymmetric(y_train, noise, random_state=None):

    """mistakes:
        0 - religion <-> 5 - politics
        1 - computers -> 4 - science
        2 - for sale <- 3 - autos/motorcylce
    """

    nb_classes = 6
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # religion <-> politics
        P[0, 0], P[0, 5] = 1. -n, n
        P[5, 0], P[5, 5] = n, 1. - n

        # computers -> science
        P[1, 1], P[1, 4] = 1. - n, n

        # for sale <- autos/motorcylce
        P[3, 3], P[3, 2] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

        return y_train, P

def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std): #, seed): 

    """
    Code snippet taken from:
    https://github.com/xiaoboxia/Part-dependent-label-noise/blob/main/tools.py
    """

    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 

    
    print("building dataset...")
    label_num = num_classes
    # np.random.seed(int(seed))
    # torch.manual_seed(int(seed))
    # torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm
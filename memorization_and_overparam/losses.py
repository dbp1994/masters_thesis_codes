import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class MAE(nn.Module):

    def __init__(self, num_class=10, reduction="none"):
        super(MAE, self).__init__()

        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(device)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps)

        if self.reduction == "none":
            return torch.sum(F.l1_loss(y_pred, y_true, reduction="none"), dim=1)
        else:
            return F.l1_loss(y_pred, y_true, reduction=self.reduction)

class MSE(nn.Module):

    def __init__(self, num_class=10, reduction="none"):
        super(MSE, self).__init__()

        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(device)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps)

        if self.reduction == "none":
            return torch.sum(F.mse_loss(y_pred, y_true, reduction="none"), dim=1)
        else:
            return F.mse_loss(y_pred, y_true, reduction=self.reduction)

class MSE_APL(nn.Module):
    def __init__(self, alpha = 0.1, beta = 1, num_class=10, reduction="none"):
        super(MSE_APL, self).__init__()
        self.reduction = reduction
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(device)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps).to(device)

        norm_const = (self.num_class*((torch.norm(y_pred, None, dim=1))**2)) + self.num_class - 2

        # norm_const = torch.clamp(norm_const, min=1e-8)

        l1 = (1./norm_const) * torch.sum(F.mse_loss(y_pred, y_true, reduction="none"), dim=1)
        ## l2 = -1. * torch.sum(y_pred * torch.log(torch.clamp(y_true, eps, 1.0)), dim=1)
        # l2 = torch.sum(F.l1_loss(y_pred, y_true, reduction="none"), dim=1)

        if self.reduction == "mean":
            return self.alpha * torch.mean(l1, dim=0) # + (self.beta * torch.mean(l2, dim=0))
        elif self.reduction == "sum":
            return self.alpha * torch.sum(l1, dim=0) # + (self.beta * torch.sum(l2, dim=0))
        else:
            return (self.alpha * l1) # + (self.beta * l2)


class L_DMI(nn.Module):

    def __init__(self, num_class=10):
        super(L_DMI, self).__init__()

        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):
        """prediction and target_label should be of size [batch_size, num_class]
        """
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(device)

        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps)

        U = torch.matmul(torch.transpose(y_true,0,1), y_pred)
        return -1.0 * torch.log(torch.abs(torch.det(U.type(torch.FloatTensor))).type(torch.FloatTensor) + 1e-3)


class GCE(nn.Module):
    """
    Implementing Generalised Cross-Entropy (GCE) Loss 
    """
    def __init__(self, q = 0.7, num_class = 10, reduction="none"):
        super(GCE, self).__init__()

        self.q = q
        self.reduction = reduction
        self.num_class = num_class
    
    def forward(self, prediction, target_label, one_hot=True):
        """
        Function to compute GCE loss.
        Usage: total_loss = GCE(target_label, prediction)
        Arguments:
            prediction : A 2d tensor of shape (batch_size, num_classes), with
                        each element in the ith row representing the 
                        probability of the corresponding class being present
                        in the ith sample.
            target_label: A 2d tensor of shape (batch_size, num_classes), with 
                        each element in the ith row representing the presence 
                        or absence of the corresponding class in the ith 
                        sample.
        """
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).to(device)

        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps)

        t_loss = (1. - torch.pow(torch.sum(y_true.type(torch.float)
                            * y_pred, dim=1), self.q)) / self.q
        
        if self.reduction == "mean":
            return torch.mean(t_loss, dim=0) 
        elif self.reduction == "sum":
            return torch.sum(t_loss, dim=0)
        else:
            return t_loss

        
def get_loss(loss_name, num_class, reduction="none", **kwargs):

    if loss_name == "cce":
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    elif loss_name == "cce_new":
        try:
            alpha = kwargs['alpha']
            beta = kwargs['beta']
        except KeyError:
            alpha = 0.1
            beta = 1
        loss_fn = CCE_new_APL(alpha=alpha, beta=beta, num_class=num_class, reduction=reduction)
    elif loss_name == "gce":
        try:
            q = kwargs['q']
        except KeyError:
            q = 0.7
        loss_fn = GCE(q=q, num_class=num_class, reduction=reduction)
    elif loss_name == "dmi":
        loss_fn = L_DMI(num_class=num_class)
    elif loss_name == "rll":
        try:
            alpha = kwargs['alpha']
        except KeyError:
            alpha = 0.01 # 0.45 # 0.45/0.5/0.6 => works well with lr = 3e-3 => ADAM
        loss_fn = RLL(alpha=alpha, num_class=num_class, reduction=reduction)
    elif loss_name == "mae":
        loss_fn = MAE(num_class=num_class, reduction=reduction)
    elif loss_name == "mse":
        loss_fn = MSE(num_class=num_class, reduction=reduction)
    elif loss_name == "norm_mse":
        try:
            alpha = kwargs['alpha']
            beta = kwargs['beta']
        except KeyError:
            alpha = 1 # 0.1
            beta = 1
        loss_fn = MSE_APL(alpha=alpha, beta=beta, num_class=num_class, reduction=reduction)
    elif loss_name == "soft_hinge":
        loss_fn = soft_hinge_mult(num_class=num_class, reduction=reduction)
    elif loss_name == "forward":
        try:
            P = kwargs['P']
        except KeyError:
            raise SystemExit("Forgot to pass the estimated transition matrix.")
        loss_fn = LC_forward(P=P, num_class=num_class, reduction=reduction)
    elif loss_name == "backward":
        try:
            P = kwargs['P']
        except KeyError:
            raise SystemExit("Forgot to pass the estimated transition matrix.")
        loss_fn = LC_backward(P=P, num_class=num_class, reduction=reduction)
    else:
        raise NotImplementedError("Loss Function Not Implemented.\n")

    return loss_fn

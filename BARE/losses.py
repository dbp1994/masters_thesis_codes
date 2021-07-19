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

class soft_hinge_mult(nn.Module):

    def __init__(self, num_class=10, reduction="none"):
        super(soft_hinge_mult, self).__init__()

        self.reduction = reduction
        self.num_class = num_class

    def forward(self, output, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(device)

        # Compute Margins for multi-class
        marg_lab = torch.sum(output*y_true, dim=1)
        marg_max, _ = torch.max(output - output*y_true, dim=1)
        marg_1 = marg_lab - marg_max
        marg_2 = marg_lab - torch.logsumexp(output, dim=1)

        idx_mask = torch.zeros(output.shape[0]).to(device)
        idx_tmp = torch.where(marg_1 >= 0)[0]
        idx_mask[idx_tmp] = 1

        loss_val = idx_mask * torch.max(1. -  marg_1, torch.zeros(output.shape[0]).to(device)
                ) + ((1 - idx_mask) * torch.max(1. -  marg_2, torch.zeros(output.shape[0]).to(device)))

        if self.reduction == "none":
            return loss_val
        elif self.reduction == "sum":
            return torch.sum(loss_val)
        elif self.reduction == "mean":
            return torch.mean(loss_val)

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

        # norm_const = torch.sum(F.mse_loss(y_pred.repeat_interleave(self.num_class, dim=0), 
        #             (torch.eye(self.num_class)).repeat(y_pred.shape[0], 1).to(device), 
        #             reduction="none"), dim=1)
        # # print("\n", norm_const.shape,"\n")
        # norm_const = torch.reshape(norm_const, (self.num_class, -1))
        # # print("\n", norm_const.shape,"\n")
        # norm_const = torch.sum(norm_const, dim=0)
        # # print("\n", norm_const.shape,"\n")

        # if torch.min(norm_const) < 1e-8:
        #     raise SystemExit("Denominator too small.\n")

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

class CCE_new_APL(nn.Module):

    def __init__(self, alpha = 0.1, beta = 1, num_class=10, reduction="none"):
        super(CCE_new_APL, self).__init__()
        self.reduction = reduction
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target_label, one_hot=True):
        
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(device)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps)

        l1 = torch.sum(y_true * torch.log(1. + y_pred), dim=1)

        norm_const = torch.sum(torch.log(1. + y_pred.repeat_interleave(self.num_class, dim=0)) *  
                    (torch.eye(self.num_class)).repeat(y_true.shape[0], 1).to(device), dim=1)
        norm_const = torch.reshape(norm_const, (self.num_class, -1))
        norm_const = torch.sum(norm_const, dim=0)

        if torch.min(norm_const) < 1e-8:
            raise SystemExit("Denominator too small.\n")

        l1 = (1./norm_const) * l1
        ## l2 = -1. * torch.sum(y_pred * torch.log(torch.clamp(y_true, eps, 1.0)), dim=1) # gives denom. < 1e-8
        l2 = torch.sum(F.l1_loss(y_pred, y_true, reduction="none"), dim=1)

        if self.reduction == "sum":
            return (self.alpha * torch.sum(l1, dim=0)) + (self.beta * torch.sum(l2, dim=0))
        elif self.reduction == "mean":
            return (self.alpha * torch.mean(l1, dim=0)) + (self.beta * torch.mean(l2, dim=0))
        else:
            return (self.alpha * l1) + (self.beta * l2)


class weighted_CCE(nn.Module):
    """
    Implementing Weighted Generalised Cross-Entropy (GCE) Loss 
    """

    def __init__(self, k=1, num_class=10, reduction="mean"):
        super(weighted_CCE, self).__init__()

        self.k = k
        self.reduction = reduction
        self.num_class= num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).to(device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)

        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        ## Compute batch statistics

        # print("pred_tmp", pred_tmp)
        # print("pred_tmp", pred_tmp.shape)

        avg_post = torch.mean(y_pred, dim=0)
        # print(avg_post)
        # print(avg_post.shape)
        avg_post = avg_post.reshape(-1, 1)
        # print(avg_post.shape)

        # med_post = torch.median(y_pred, dim=0,keepdim=True).values
        # # # print(f"\nmed_post: {med_post}\n")
        # med_post = med_post.reshape(-1, 1)
        # # # print(f"\nmed_post: {med_post.shape}\n")
        # med_post_ref = torch.matmul(y_true.type(torch.float), med_post)

        std_post = torch.std(y_pred, dim=0)
        # print(std_post)
        std_post = std_post.reshape(-1, 1)
        # print(std_post.shape)

        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        # print("avg_post_ref", avg_post_ref)
        # print("avg_post_ref", avg_post_ref.shape)

        std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
        # print("std_post_ref", std_post_ref)
        # print("std_post_ref", std_post_ref.shape)

        # pred_prun = torch.where((torch.abs(pred_tmp - avg_post_ref) <= std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
        # pred_prun = torch.where((pred_tmp >= avg_post_ref), pred_tmp, torch.zeros_like(pred_tmp))

        pred_prun = torch.where((pred_tmp - avg_post_ref >= self.k * std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
        # pred_prun = torch.where((pred_tmp >= med_post_ref), pred_tmp, torch.zeros_like(pred_tmp))


        # prun_idx will tell us which examples are 
        # 'trustworthy' for the given batch
        prun_idx = torch.where(pred_prun != 0.)[0]

        # print("pred_prun", pred_prun)
        # print("pred_prun", pred_prun.shape)
        # print("prun_idx", prun_idx)
        # print("prun_idx", prun_idx.shape)

        # input("<ENTER>\n")

        # loss_fn = nn.CrossEntropyLoss(reduction=self.reduction)
        # weighted_loss=loss_fn(torch.index_select(y_true, dim=0, prun_idx), 
        #                 torch.index_select(y_pred, dim=0, prun_idx))

        if len(prun_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, prun_idx), dim=1)
            weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, prun_idx), 
                            prun_targets, reduction=self.reduction)
        else:
            weighted_loss = F.cross_entropy(prediction, target_label)

        return weighted_loss, prun_idx




class weighted_RLL(nn.Module):
    """
    Implementing Weighted Generalised Cross-Entropy (GCE) Loss 
    """

    def __init__(self, k=1, alpha=0.45, num_class=10, reduction="mean"):
        super(weighted_RLL, self).__init__()

        self.k = k
        self.alpha = torch.Tensor([alpha]).to(device)
        self.reduction = reduction
        self.num_class= num_class
        

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).to(device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)

        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        ## Compute batch statistics

        avg_post = torch.mean(y_pred, dim=0)
        avg_post = avg_post.reshape(-1, 1)

        std_post = torch.std(y_pred, dim=0)
        std_post = std_post.reshape(-1, 1)

        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        # print("avg_post_ref", avg_post_ref)
        # print("avg_post_ref", avg_post_ref.shape)

        std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
        # print("std_post_ref", std_post_ref)
        # print("std_post_ref", std_post_ref.shape)

        # pred_prun = torch.where((torch.abs(pred_tmp - avg_post_ref) <= std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
        pred_prun = torch.where((pred_tmp - avg_post_ref >= self.k * std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
        # pred_prun = torch.where((pred_tmp >= med_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
        # pred_prun = torch.where((pred_tmp >= avg_post_ref), pred_tmp, torch.zeros_like(pred_tmp))

        # prun_idx will tell us which examples are 
        # 'trustworthy' for the given batch
        prun_idx = torch.where(pred_prun != 0.)[0]

        if len(prun_idx) != 0:
            y_t = (((1.-torch.index_select(y_true, 0, prun_idx))/(self.num_class - 1))*torch.log(self.alpha + torch.index_select(y_pred, 0, 
                    prun_idx))) - (torch.index_select(y_true, 0, prun_idx)*torch.log(self.alpha + torch.index_select(y_pred, 0, prun_idx))) \
                    + torch.index_select(y_true, 0, prun_idx)*(torch.log(self.alpha + 1) - torch.log(self.alpha))
        
        else:
            y_t = (((1.-y_true)/(self.num_class - 1))*torch.log(self.alpha + y_pred)) - (y_true*torch.log(self.alpha + y_pred)) \
                    + y_true*(torch.log(self.alpha + 1) - torch.log(self.alpha))

        temp = torch.sum(y_t, dim = 1)        

        if self.reduction == "none":
            return temp
        elif self.reduction == "mean":
            return torch.mean(temp, dim=0)
        elif self.reduction == "sum":
            return torch.sum(temp, dim=0)

class RLL(nn.Module):

    def __init__(self, alpha=0.45, num_class=10, reduction="none"):
        super(RLL, self).__init__()
        
        self.alpha = torch.Tensor([alpha]).to(device)
        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).to(device)

        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps)

        y_t = (((1.-y_true)/(self.num_class - 1))*torch.log(self.alpha + y_pred)) - (y_true*torch.log(self.alpha + y_pred)) \
                    + y_true*(torch.log(self.alpha + 1) - torch.log(self.alpha))
        
        temp = torch.sum(y_t, dim = 1)

        if self.reduction == "none":
            return temp
        elif self.reduction == "mean":
            return torch.mean(temp, dim=0)
        elif self.reduction == "sum":
            return torch.sum(temp, dim=0)

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

class GCE_APL(nn.Module):
    def __init__(self, alpha = 1, beta = 1, num_class=10, reduction="none"):
        super(GCE_APL, self).__init__()
        self.reduction = reduction
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(device)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, eps, 1-eps)

        gce_loss = GCE()

        norm_const = torch.sum(gce_loss(prediction.repeat_interleave(self.num_class, dim=0), 
                    (torch.eye(self.num_class)).repeat(y_pred.shape[0], 1).to(device), 
                    reduction="none"), dim=1)
        norm_const = torch.reshape(norm_const, (self.num_class, -1))
        norm_const = 1.* self.num_class - torch.sum(norm_const, dim=0)

        if torch.min(norm_const) < 1e-8:
            raise SystemExit("Denominator too small.\n")

        l1 = (1./norm_const) * torch.sum(gce_loss(prediction, y_true, reduction="none"), dim=1)
        # l2 = -1. * torch.sum(y_pred * torch.log(torch.clamp(y_true, eps, 1.0)), dim=1)
        l2 = torch.sum(F.l1_loss(y_pred, y_true, reduction="none"), dim=1)

        if self.reduction == "mean":
            return self.alpha * torch.mean(l1, dim=0) + (self.beta * torch.mean(l2, dim=0))
        elif self.reduction == "sum":
            return self.alpha * torch.sum(l1, dim=0) + (self.beta * torch.sum(l2, dim=0))
        else:
            return (self.alpha * l1) + (self.beta * l2)


class weighted_GCE(nn.Module):
    def __init__(self, k=1, q=0.7, num_class=10, reduction="mean"):
        super(weighted_GCE, self).__init__()

        self.k = k
        self.q = q
        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_class=self.num_class).to(device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)

        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        ## Compute batch statistics

        # print("pred_tmp", pred_tmp)
        # print("pred_tmp", pred_tmp.shape)

        avg_post = torch.mean(y_pred, axis=0)
        # print(avg_post)
        # print(avg_post.shape)
        avg_post = avg_post.reshape(-1, 1)
        # print(avg_post.shape)

        std_post = torch.std(y_pred, dim=0)
        # print(std_post)
        std_post = std_post.reshape(-1, 1)
        # print(std_post.shape)

        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        # print("avg_post_ref", avg_post_ref)
        # print("avg_post_ref", avg_post_ref.shape)

        std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
        # print("std_post_ref", std_post_ref)
        # print("std_post_ref", std_post_ref.shape)

        # pred_prun = torch.where((torch.abs(pred_tmp - avg_post_ref) <= std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
        pred_prun = torch.where((pred_tmp - avg_post_ref >= self.k * std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
        # pred_prun = torch.where((pred_tmp >= avg_post_ref), pred_tmp, torch.zeros_like(pred_tmp))

        # prun_idx will tell us which examples are 
        # 'trustworthy' for the given batch
        prun_idx = torch.where(pred_prun != 0.)[0]

        # print("pred_prun", pred_prun)
        # print("pred_prun", pred_prun.shape)
        # print("prun_idx", prun_idx)
        # print("prun_idx", prun_idx.shape)

        # input("<ENTER>\n")

        # loss_fn = nn.CrossEntropyLoss(reduction=self.reduction)
        # weighted_loss=loss_fn(torch.index_select(y_true, dim=0, prun_idx), 
        #                 torch.index_select(y_pred, dim=0, prun_idx))

        # prun_targets = torch.argmax(torch.index_select(y_true, 0, prun_idx), dim=1)

        if len(prun_idx) != 0:
            weighted_loss = (1 - torch.pow(torch.sum(torch.index_select(y_true, 0, 
                                    prun_idx).type(torch.float)
                                * torch.index_select(y_pred, 0, 
                                prun_idx), dim=1), self.q)) / self.q
        else:
            weighted_loss = (1 - torch.pow(torch.sum(y_true.type(torch.float)
                                * y_pred, self.q))) / self.q
        
        if self.reduction == "mean":
            return torch.mean(weighted_loss, dim=0) 
        elif self.reduction == "sum":
            return torch.sum(weighted_loss, dim=0)
        else:
            return weighted_loss


class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).to(self.device).random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).to(self.device)
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss


class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NormalizedFocalLoss(torch.nn.Module):

    def __init__(self, scale=1.0, gamma=0, num_class=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_class = num_class
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class MarginLoss(nn.Module):
    """
        Class to compute margin loss.
    """

    def __init__(self, num_class=10):
        super(MarginLoss, self).__init__()
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):
        """
            Function to compute margin loss.
            Usage: total_loss = margin_loss(prediction, target_label).
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

        present_loss = F.relu(0.9 - y_pred) ** 2
        absent_loss = F.relu(y_pred - 0.1) ** 2

        margin_loss = y_true * present_loss + 0.5 * (1. - y_true) * absent_loss
        margin_loss = margin_loss.sum()

        return margin_loss

class LC_forward(nn.Module):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach.
    """
    def __init__(self, P, num_class=10, reduction="none"):
        super(LC_forward, self).__init__()

        self.reduction = reduction
        self.num_class= num_class
        self.P = P
        
    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).to(device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)

        y_pred /= torch.sum(y_pred, dim=-1, keepdims=True)
        y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
        temp = -torch.sum(y_true * torch.log(torch.matmul(y_pred, self.P)), dim=-1)

        if self.reduction == "none":
            return temp
        elif self.reduction == "mean":
            return torch.mean(temp, dim=0)
        elif self.reduction == "sum":
            return torch.sum(temp, dim=0)

class LC_backward(nn.Module):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach.
    """
    def __init__(self, P, num_class=10, reduction="none"):
        super(LC_backward, self).__init__()

        self.reduction = reduction
        self.num_class= num_class

        self.P_inv = torch.linalg.inv(P)
        # self.P_inv = torch.Tensor(P_inv).to(device)
        
    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).to(device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)

        y_pred /= torch.sum(y_pred, dim=-1, keepdims=True)
        y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
        temp = -torch.sum((y_true * self.P_inv) * torch.log(y_pred), dim=-1)

        if self.reduction == "none":
            return temp
        elif self.reduction == "mean":
            return torch.mean(temp, dim=0)
        elif self.reduction == "sum":
            return torch.sum(temp, dim=0)


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


# def joint_optimization_loss(y_true, y_pred):
#     """
#     2018 - cvpr - Joint optimization framework for learning with noisy labels.
#     """
#     y_pred_avg = K.mean(y_pred, axis=0)
#     p = np.ones(10, dtype=np.float32) / 10.
#     l_p = - K.sum(K.log(y_pred_avg) * p)
#     l_e = K.categorical_crossentropy(y_pred, y_pred)
#     return K.categorical_crossentropy(y_true, y_pred) + 1.2 * l_p + 0.8 * l_e

# def boot_soft(y_true, y_pred):
#     """
#     2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
#     """
#     beta = 0.95

#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     return -K.sum((beta * y_true + (1. - beta) * y_pred) *
#                   K.log(y_pred), axis=-1)


# def boot_hard(y_true, y_pred):
#     """
#     2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
#     """
#     beta = 0.8

#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
#     return -K.sum((beta * y_true + (1. - beta) * pred_labels) *
#            K.log(y_pred), axis=-1)
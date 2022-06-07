from math import cos, pi
from re import S
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        n = float(n)
        self.sum += val * n
        self.count += n
    
    def avg(self):
        return (self.sum / self.count)


def statistics(pred, y, thresh):
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list


def BP4D_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}
    return infostr

def DISFA_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    return infostr

def SAW2_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU15: {:.2f} AU23: {:.2f} AU24: {:.2f} AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}
    return infostr

def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class image_train(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img

class image_train_saw2(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, img, flip):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            SetFlip(flip),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img

class image_test_saw2(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


class image_test(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model

EPS = 1e-8

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y, mask):

        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        loss = loss * mask
        return -loss.sum() / (mask.sum() + EPS)

class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__() 
        self.loss = MaskNegativeCCCLoss()

    def forward(self, x, y, mask):
        loss1 = self.loss(x[:, 0], y[:, 0], mask) + self.loss(x[:, 1], y[:, 1], mask)
        return loss1

class MaskedCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(MaskedCELoss, self).__init__() 
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight, ignore_index=ignore_index)
    
    def forward(self, x, y, mask):
        loss = self.ce(x, y)
        loss = loss.mean(dim=-1)
        loss = loss * mask
        return loss.sum() / (mask.sum() + EPS)

class NegativeCCCLoss(nn.Module):
    def __init__(self, digitize_num=20, range=[-1, 1], weight=None):
        super(NegativeCCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        self.weight = weight
        if self.digitize_num >1:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = torch.as_tensor(bins, dtype = torch.float32).cuda().view((1, -1))
    def forward(self, x, y): 
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1) # expectation
        x = x.view(-1)
        if self.weight is None:
            vx = x - torch.mean(x) 
            vy = y - torch.mean(y) 
            rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + EPS)
            x_m = torch.mean(x)
            y_m = torch.mean(y)
            x_s = torch.std(x)
            y_s = torch.std(y)
            ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
        else:
            rho = weighted_correlation(x, y, self.weight)
            x_var = weighted_covariance(x, x, self.weight)
            y_var = weighted_covariance(y, y, self.weight)
            x_mean = weighted_mean(x, self.weight)
            y_mean = weighted_mean(y, self.weight)
            ccc = 2*rho*torch.sqrt(x_var)*torch.sqrt(y_var)/(x_var + y_var + torch.pow(x_mean - y_mean, 2) +EPS)
        return 1-ccc

class MaskNegativeCCCLoss(nn.Module):
    def __init__(self):
        super(MaskNegativeCCCLoss, self).__init__()
    def forward(self, x, y, m):
        y = y.view(-1)
        x = x.view(-1)
        x = x * m
        y = y * m
        N = torch.sum(m)
        x_m = torch.sum(x) / N
        y_m = torch.sum(x) / N
        vx = (x - x_m) * m
        vy = (y - y_m) * m
        ccc = 2*torch.dot(vx, vy) / (torch.dot(vx, vx) + torch.dot(vy, vy) + N * torch.pow(x_m - y_m, 2) + EPS)
        return 1-ccc

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def EX_metric(y, yhat):
    i = np.argmax(yhat, axis=1)
    yhat = np.zeros(yhat.shape)
    yhat[np.arange(len(i)), i] = 1

    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    if not len(yhat.shape) == 1:
        if yhat.shape[1] == 1:
            yhat = yhat.reshape(-1)
        else:
            yhat = np.argmax(yhat, axis=-1)

    return f1_score(y, yhat, average='macro')


def VA_metric(y, yhat):
    avg_ccc = float(CCC_score(y[:,0], yhat[:,0]) + CCC_score(y[:,1], yhat[:,1])) / 2
    return avg_ccc


def AU_metric(y, yhat, thresh=0.5):
    yhat = (yhat >= thresh)
    N, label_size = y.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(y[:, i], yhat[:, i])
        f1s.append(f1)
    return np.mean(f1s)

def ConvertNum2Weight(x, y, z):
    x, y, z = float(x), float(y), float(z)
    if x!=0:
        x = 1. / x
    if y!=0:
        y = 1. / y
    if z!=0:
        z = 1. / z
    if x==0 and y==0 and z==0:
        return 0,0,0
    else:
        a = (x + y + z) / 3.
        return x / a, y / a, z / a
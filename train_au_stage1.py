import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.ANFL import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'SAW2':
        trainset = SAW2(conf.dataset_path, train=True, transform=image_train_saw2(img_size=224), stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = SAW2(conf.dataset_path, train=False, transform=image_test_saw2(img_size=224), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)


# Train
def train(conf,net,train_loader,optimizer,epoch,criteria):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs, y_va, y_ex, y_au, mask_va, mask_ex, mask_au) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        y_va = y_va.float()
        # y_ex = y_ex.float()
        y_au = y_au.float()
        mask_va = mask_va.float().unsqueeze(-1)
        mask_ex = mask_ex.float().unsqueeze(-1)
        mask_au = mask_au.float().unsqueeze(-1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            y_va = y_va.cuda()
            y_ex = y_ex.cuda()
            y_au = y_au.cuda()
            mask_va = mask_va.cuda()
            mask_ex = mask_ex.cuda()
            mask_au = mask_au.cuda()
        optimizer.zero_grad()
        yhat_va, yhat_ex, yhat_au = net(inputs)
        y_va = mask_va * y_va
        y_ex = mask_ex * y_ex
        y_au = mask_au * y_au
        yhat_va = mask_va * yhat_va
        yhat_ex = mask_ex * yhat_ex
        yhat_au = mask_au * yhat_au
        loss = criteria['VA'](yhat_va, y_va) + criteria['EX'](yhat_ex, y_ex) + criteria['AU'](yhat_au, y_au)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg


# Val
def val(net,val_loader,criteria):
    losses = AverageMeter()
    net.eval()
    all_y_va = None
    all_y_ex = None
    all_y_au = None
    all_yhat_va = None
    all_yhat_ex = None
    all_yhat_au = None
    for batch_idx, (inputs, y_va, y_ex, y_au, mask_va, mask_ex, mask_au) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            y_va = y_va.float()
            # y_ex = y_ex.float()
            y_au = y_au.float()
            mask_va = mask_va.float().unsqueeze(-1)
            mask_ex = mask_ex.float().unsqueeze(-1)
            mask_au = mask_au.float().unsqueeze(-1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                y_va = y_va.cuda()
                y_ex = y_ex.cuda()
                y_au = y_au.cuda()
                mask_va = mask_va.cuda()
                mask_ex = mask_ex.cuda()
                mask_au = mask_au.cuda()
            yhat_va, yhat_ex, yhat_au = net(inputs)
            y_va = mask_va * y_va
            y_ex = mask_ex * y_ex
            y_au = mask_au * y_au
            yhat_va = mask_va * yhat_va
            yhat_ex = mask_ex * yhat_ex
            yhat_au = mask_au * yhat_au
            loss = criteria['VA'](yhat_va, y_va) + criteria['EX'](yhat_ex, y_ex) + criteria['AU'](yhat_au, y_au)
            losses.update(loss.data.item(), inputs.size(0))

            if all_y_va == None:
                all_y_va = y_va.clone()
                all_y_ex = y_ex.clone()
                all_y_au = y_au.clone()
                all_yhat_va = yhat_va.clone()
                all_yhat_ex = yhat_ex.clone()
                all_yhat_au = yhat_au.clone()
            else:
                all_y_va = torch.cat((all_y_va, y_va), 0)
                all_y_ex = torch.cat((all_y_ex, y_ex), 0)
                all_y_au = torch.cat((all_y_au, y_au), 0)
                all_yhat_va = torch.cat((all_yhat_va, yhat_va), 0)
                all_yhat_ex = torch.cat((all_yhat_ex, yhat_ex), 0)
                all_yhat_au = torch.cat((all_yhat_au, yhat_au), 0)

    all_y_va = all_y_va.cpu()
    all_y_ex = all_y_ex.cpu()
    all_y_au = all_y_au.cpu()
    all_yhat_va = all_yhat_va.cpu()
    all_yhat_ex = all_yhat_ex.cpu()
    all_yhat_au = all_yhat_au.cpu()
    va_metrics = VA_metric(all_y_va, all_yhat_va)
    ex_metrics = EX_metric(all_y_ex, all_yhat_ex)
    au_metrics = AU_metric(all_y_au, all_yhat_au)
    performance = va_metrics + ex_metrics + au_metrics
    
    # mean_acc, acc_list = calc_acc(au_statistics_list)
    return losses.avg, va_metrics, ex_metrics, au_metrics, performance


def main(conf):
    start_epoch = 0
    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join('train_weight.txt')))

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    criteria = {}
    criteria['EX'] = CrossEntropyLoss
    criteria['VA'] = RegressionLoss
    criteria['AU'] = WeightedAsymmetricLoss(weight=train_weight)
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf,net,train_loader,optimizer,epoch,criteria)
        val_loss, val_va_metrics, val_ex_metrics, val_au_metrics, val_performance = val(net, val_loader, criteria)

        # log
        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_va_metrics {:.2f}   val_ex_metrics {:.2f}  val_au_metrics {:.2f}  val_performance {:.2f}'
                .format(epoch + 1, train_loss, val_loss, val_va_metrics, 100.* val_ex_metrics, 100.* val_au_metrics, val_performance)}

        logging.info(infostr)

        # save checkpoints
        if (epoch+1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold) + '.pth'))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)


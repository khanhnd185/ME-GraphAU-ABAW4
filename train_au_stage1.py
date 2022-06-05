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
        trainldr = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        validset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        validldr = DataLoader(validset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        trainldr = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        validset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        validldr = DataLoader(validset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'SAW2':
        trainset = SAW2(conf.dataset_path, train=True, transform=image_train_saw2(img_size=224), stage = 1)
        trainldr = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        validset = SAW2(conf.dataset_path, train=False, transform=image_test_saw2(img_size=224), stage = 1)
        validldr = DataLoader(validset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return trainldr, validldr

def train(conf, net, trainldr, optimizer, epoch, criteria):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for batch_idx, (inputs, y_va, y_ex, y_au, mask_va, mask_ex, mask_au) in enumerate(tqdm(trainldr)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        y_va = y_va.float()
        y_au = y_au.float()
        mask_va = mask_va.float().unsqueeze(-1)
        mask_au = mask_au.float()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            y_va = y_va.cuda()
            y_ex = y_ex.cuda()
            y_au = y_au.cuda()
            mask_va = mask_va.cuda()
            mask_au = mask_au.cuda()
        optimizer.zero_grad()
        yhat_va, yhat_ex, yhat_au = net(inputs)
        loss = criteria['VA'](yhat_va, y_va, mask_va) + criteria['EX'](yhat_ex, y_ex) + criteria['AU'](yhat_au, y_au, mask_au)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg

def val(net, validldr, criteria):
    losses = AverageMeter()
    net.eval()
    all_y_va = None
    all_y_ex = None
    all_y_au = None
    all_yhat_va = None
    all_yhat_ex = None
    all_yhat_au = None
    for batch_idx, (inputs, y_va, y_ex, y_au, mask_va, mask_ex, mask_au) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            y_va = y_va.float()
            y_au = y_au.float()
            mask_va = mask_va.float().unsqueeze(-1)
            mask_au = mask_au.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                y_va = y_va.cuda()
                y_ex = y_ex.cuda()
                y_au = y_au.cuda()
                mask_va = mask_va.cuda()
                mask_au = mask_au.cuda()
            yhat_va, yhat_ex, yhat_au = net(inputs)
            loss = criteria['VA'](yhat_va, y_va, mask_va) + criteria['EX'](yhat_ex, y_ex) + criteria['AU'](yhat_au, y_au, mask_au)
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
    return losses.avg, va_metrics, ex_metrics, au_metrics, performance

def main(conf):
    start_epoch = 0
    train_loader, valid_loader = get_dataloader(conf)
    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)

    valid_au_weight = torch.from_numpy(np.loadtxt(os.path.join('valid_weight_au.txt')))
    valid_ex_weight = torch.from_numpy(np.loadtxt(os.path.join('valid_weight_ex.txt')))
    train_au_weight = torch.from_numpy(np.loadtxt(os.path.join('train_weight_au.txt')))
    train_ex_weight = torch.from_numpy(np.loadtxt(os.path.join('train_weight_ex.txt')))

    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    train_loss = {}
    valid_loss = {}
    train_loss['VA'] = RegressionLoss
    valid_loss['VA'] = RegressionLoss
    train_loss['AU'] = WeightedAsymmetricLoss(weight=train_au_weight)
    valid_loss['AU'] = WeightedAsymmetricLoss(weight=valid_au_weight)
    train_loss['EX'] = nn.CrossEntropyLoss(weight=train_ex_weight, ignore_index=-1)
    valid_loss['EX'] = nn.CrossEntropyLoss(weight=valid_ex_weight, ignore_index=-1)

    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch, conf.epochs, lr))
        train_loss = train(conf, net, train_loader, optimizer, epoch, train_loss)
        val_loss, val_va_metrics, val_ex_metrics, val_au_metrics, val_performance = val(net, valid_loader, valid_loss)

        infostr = {'Epoch: {} \
                    train_loss: {:.5f} \
                    val_loss: {:.5f} \
                    val_va_metrics: {:.2f} \
                    val_ex_metrics: {:.2f} \
                    val_au_metrics: {:.2f} \
                    val_performance: {:.2f}'
                    .format(epoch,
                            train_loss,
                            val_loss,
                            val_va_metrics,
                            val_ex_metrics,
                            val_au_metrics,
                            val_performance)}

        logging.info(infostr)

        if (epoch+1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch) + '_model_fold' + str(conf.fold) + '.pth'))

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold) + '.pth'))

if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)

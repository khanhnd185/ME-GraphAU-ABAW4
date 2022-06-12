import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import math

from model.MEFL import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 2)
        trainldr = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        validset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2)
        validldr = DataLoader(validset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 2)
        trainldr = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        validset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2)
        validldr = DataLoader(validset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'SAW2':
        trainset = SAW2(conf.dataset_path, train=True, transform=image_train_saw2(img_size=224), stage = 2)
        trainldr = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        validset = SAW2(conf.dataset_path, train=False, transform=image_test_saw2(img_size=224), stage = 2)
        validldr = DataLoader(validset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return trainldr, validldr

def train(conf, net, trainldr, optimizer, epoch, criteria, weight):
    total_losses = AverageMeter()
    va_losses = AverageMeter()
    ex_losses = AverageMeter()
    au_losses = AverageMeter()
    edge_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for batch_idx, (inputs, y_va, y_ex, y_au, mask_va, mask_ex, mask_au, relations) in enumerate(tqdm(trainldr)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        y_va = y_va.float()
        y_au = y_au.float()
        y_ex = y_ex.long()
        mask_va = mask_va.float()
        mask_ex = mask_ex.float()
        mask_au = mask_au.float()
        relations = relations.long()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            y_va = y_va.cuda()
            y_ex = y_ex.cuda()
            y_au = y_au.cuda()
            mask_va = mask_va.cuda()
            mask_ex = mask_ex.cuda()
            mask_au = mask_au.cuda()
            relations = relations.cuda()
        optimizer.zero_grad()
        yhat_va, yhat_ex, yhat_au, outputs_relation = net(inputs)
        relations = mask_au.unsqueeze(-1).long() * relations
        outputs_relation = mask_au.unsqueeze(-1).unsqueeze(-1).repeat(1, 144 ,4) * outputs_relation
        loss_va = weight[0] * criteria['VA'](yhat_va, y_va, mask_va)
        loss_ex = weight[1] * criteria['EX'](yhat_ex, y_ex, mask_ex)
        loss_au = weight[2] * criteria['AU'](yhat_au, y_au, mask_au)
        loss = loss_va + loss_ex + loss_au
        edge_loss = criteria['REL'](outputs_relation.view(-1,4), relations.view(-1), mask_au)
        total_loss = loss + weight[2] * conf.lam * edge_loss
        total_loss.backward()
        optimizer.step()
        total_losses.update(total_loss.data.item(), inputs.size(0))
        va_losses.update(loss_va.data.item(), inputs.size(0))
        ex_losses.update(loss_ex.data.item(), inputs.size(0))
        au_losses.update(loss_au.data.item(), inputs.size(0))
        edge_losses.update(edge_loss.data.item(), inputs.size(0))

    return total_losses.avg(), va_losses.avg(), ex_losses.avg(), au_losses.avg(), edge_losses.avg()

def val(net, validldr, criteria, weight):
    total_losses = AverageMeter()
    va_losses = AverageMeter()
    ex_losses = AverageMeter()
    au_losses = AverageMeter()
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
            y_ex = y_ex.long()
            mask_va = mask_va.float()
            mask_ex = mask_ex.float()
            mask_au = mask_au.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                y_va = y_va.cuda()
                y_ex = y_ex.cuda()
                y_au = y_au.cuda()
                mask_va = mask_va.cuda()
                mask_ex = mask_ex.cuda()
                mask_au = mask_au.cuda()
            yhat_va, yhat_ex, yhat_au, _ = net(inputs)
            loss_va = weight[0] * criteria['VA'](yhat_va, y_va, mask_va)
            loss_ex = weight[1] * criteria['EX'](yhat_ex, y_ex, mask_ex)
            loss_au = weight[2] * criteria['AU'](yhat_au, y_au, mask_au)
            total_loss = loss_va + loss_ex + loss_au
            total_losses.update(total_loss.data.item(), inputs.size(0))
            va_losses.update(loss_va.data.item(), inputs.size(0))
            ex_losses.update(loss_ex.data.item(), inputs.size(0))
            au_losses.update(loss_au.data.item(), inputs.size(0))

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
    all_y_va = all_y_va.cpu().numpy()
    all_y_ex = all_y_ex.cpu().numpy()
    all_y_au = all_y_au.cpu().numpy()
    all_yhat_va = all_yhat_va.cpu().numpy()
    all_yhat_ex = all_yhat_ex.cpu().numpy()
    all_yhat_au = all_yhat_au.cpu().numpy()
    va_index = [i for i, x in enumerate(all_y_va) if -5 not in x]
    ex_index = [i for i, x in enumerate(all_y_ex) if -1 != x]
    au_index = [i for i, x in enumerate(all_y_au) if -1 not in x]
    all_y_va = np.array([all_y_va[i] for i in va_index])
    all_y_ex = np.array([all_y_ex[i] for i in ex_index])
    all_y_au = np.array([all_y_au[i] for i in au_index])
    all_yhat_va = np.array([all_yhat_va[i] for i in va_index])
    all_yhat_ex = np.array([all_yhat_ex[i] for i in ex_index])
    all_yhat_au = np.array([all_yhat_au[i] for i in au_index])
    va_metrics = VA_metric(all_y_va, all_yhat_va)
    ex_metrics = EX_metric(all_y_ex, all_yhat_ex)
    au_metrics = AU_metric(all_y_au, all_yhat_au)
    performance = va_metrics + ex_metrics + au_metrics
    return total_losses.avg(), va_losses.avg(), ex_losses.avg(), au_losses.avg(), va_metrics, ex_metrics, au_metrics, performance

def main(conf):
    start_epoch = 0
    train_loader, valid_loader = get_dataloader(conf)
    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc)

    valid_weight = torch.from_numpy(np.loadtxt(os.path.join('valid_weight.txt')))
    train_weight = torch.from_numpy(np.loadtxt(os.path.join('train_weight.txt')))
    valid_weight = valid_weight.float()
    train_weight = train_weight.float()
    valid_au_weight = torch.from_numpy(np.loadtxt(os.path.join('valid_weight_au.txt')))
    valid_ex_weight = torch.from_numpy(np.loadtxt(os.path.join('valid_weight_ex.txt')))
    train_au_weight = torch.from_numpy(np.loadtxt(os.path.join('train_weight_au.txt')))
    train_ex_weight = torch.from_numpy(np.loadtxt(os.path.join('train_weight_ex.txt')))
    valid_ex_weight = valid_ex_weight.float()
    train_ex_weight = train_ex_weight.float()

    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        valid_weight = valid_weight.cuda()
        train_weight = train_weight.cuda()
        train_au_weight = train_au_weight.cuda()
        train_ex_weight = train_ex_weight.cuda()
        valid_au_weight = valid_au_weight.cuda()
        valid_ex_weight = valid_ex_weight.cuda()

    train_criteria = {}
    valid_criteria = {}
    train_criteria['VA'] = RegressionLoss()
    valid_criteria['VA'] = RegressionLoss()
    train_criteria['AU'] = WeightedAsymmetricLoss(weight=train_au_weight)
    valid_criteria['AU'] = WeightedAsymmetricLoss(weight=valid_au_weight)
    train_criteria['EX'] = MaskedCELoss(weight=train_ex_weight, ignore_index=-1)
    valid_criteria['EX'] = MaskedCELoss(weight=valid_ex_weight, ignore_index=-1)
    train_criteria['REL'] = MaskedCELoss(ignore_index=-3)
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    best_performance = 0.0
    best_val_loss = 100.0
    epoch_from_last_improvement = 0

    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch, conf.epochs, lr))
        train_loss, train_va_loss, train_ex_loss, train_au_loss, edge_loss = train(conf, net, train_loader, optimizer, epoch, train_criteria, train_weight)
        val_loss, val_va_loss, val_ex_loss, val_au_loss, val_va_metrics, val_ex_metrics, val_au_metrics, val_performance = val(net, valid_loader, valid_criteria, valid_weight)

        infostr = {'Epoch: {} \
        train_loss: {:.5f} \
        train_va_loss: {:.5f} \
        train_ex_loss: {:.5f} \
        train_au_loss: {:.5f} \
        edge_loss: {:.5f}'
                .format(epoch,
                        train_loss,
                        train_va_loss,
                        train_ex_loss,
                        train_au_loss,
                        edge_loss)}
        logging.info(infostr)

        infostr = {'Epoch: {} \
        val_loss: {:.5f} \
        val_va_loss: {:.5f} \
        val_ex_loss: {:.5f} \
        val_au_loss: {:.5f}'
                .format(epoch,
                        val_loss,
                        val_va_loss,
                        val_ex_loss,
                        val_au_loss)}
        logging.info(infostr)

        infostr = {'Epoch: {} \
        val_va_metrics: {:.2f} \
        val_ex_metrics: {:.2f} \
        val_au_metrics: {:.2f} \
        val_performance: {:.2f}'
                .format(epoch,
                        val_va_metrics,
                        val_ex_metrics,
                        val_au_metrics,
                        val_performance)}
        logging.info(infostr)

        if val_loss <= best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_performance': val_performance,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'best_val_loss.pth'))
            best_val_loss = val_loss

        if val_performance >= best_performance:
            checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_performance': val_performance,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'best_val_perform.pth'))
            best_performance = val_performance
            epoch_from_last_improvement = 0
        else:
            epoch_from_last_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model.pth'))

        if conf.early_stop != None:
            if epoch_from_last_improvement >= conf.early_stop:
                print('No improvement for ' + str(epoch_from_last_improvement) + ' epoches. Stop training')
                break

if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)

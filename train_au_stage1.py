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
    for batch_idx, (inputs, y_va, y_expr, y_au, mask_va, mask_expr, mask_au) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        y_va, y_expr, y_au, mask_va, mask_expr, mask_au = y_va.float(), y_expr.float(), y_au.float(), mask_va.float(), mask_expr.float(), mask_au.float()
        if torch.cuda.is_available():
            inputs, y_va, y_expr, y_au, mask_va, mask_expr, mask_au = inputs.cuda(), y_va.cuda(), y_expr.cuda(), y_au.cuda(), mask_va.cuda(), mask_expr.cuda(), mask_au.cuda()
        optimizer.zero_grad()
        yhat_va, yhat_expr, yhat_au = net(inputs)
        loss = mask_va * criteria['VA'](yhat_va, y_va) + mask_expr * criteria['VA'](yhat_expr, y_expr) + mask_au * criteria['VA'](yhat_au, y_au)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg


# Val
def val(net,val_loader,criteria):
    losses = AverageMeter()
    net.eval()
    au_statistics_list = None
    ex_statistics_list = None
    for batch_idx, (inputs, y_va, y_expr, y_au, mask_va, mask_expr, mask_au) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            y_va = y_va.float()
            y_expr = y_expr.float()
            y_au = y_au.float()
            mask_va = mask_va.float()
            mask_expr = mask_expr.float()
            mask_au = mask_au.float()
            if torch.cuda.is_available():
                y_va = y_va.cuda()
                y_expr = y_expr.cuda()
                y_au = y_au.cuda()
                mask_va = mask_va.cuda()
                mask_expr = mask_expr.cuda()
                mask_au = mask_au.cuda()
            yhat_va, yhat_expr, yhat_au = net(inputs)
            loss = mask_va * criteria['VA'](yhat_va, y_va) + mask_expr * criteria['VA'](yhat_expr, y_expr) + mask_au * criteria['VA'](yhat_au, y_au)
            losses.update(loss.data.item(), inputs.size(0))

            au_update_list = statistics(yhat_au, y_au.detach(), 0.5)
            au_statistics_list = update_statistics_list(au_statistics_list, au_update_list)

    au_mean_f1_score, _ = calc_f1_score(au_statistics_list)
    ex_mean_f1_score, _ = calc_f1_score(ex_statistics_list)
    performance = au_mean_f1_score + ex_mean_f1_score 
    
    # mean_acc, acc_list = calc_acc(au_statistics_list)
    return losses.avg, au_mean_f1_score, ex_mean_f1_score, mean_acc, performance


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
    criteria['EXPR'] = CrossEntropyLoss
    criteria['VA'] = RegressionLoss
    criteria['AU'] = WeightedAsymmetricLoss(weight=train_weight)
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf,net,train_loader,optimizer,epoch,criteria)
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criteria)

        # log
        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                .format(epoch + 1, train_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}

        logging.info(infostr)
        infostr = {'F1-score-list:'}
        logging.info(infostr)
        infostr = SAW2_infolist(val_f1_score)
        logging.info(infostr)
        infostr = {'Acc-list:'}
        logging.info(infostr)
        infostr = SAW2_infolist(val_acc)
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


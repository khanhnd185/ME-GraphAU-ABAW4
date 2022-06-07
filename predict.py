import os
import torch
import torch.nn as nn
from model.MEFL import MEFARG
from dataset import SAW2
from utils import *
import argparse

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
    parser.add_argument('--arc', '-a', type=str, help='Path to input image')
    parser.add_argument('--img', '-i', type=str, help='Backbone')
    parser.add_argument('--model', '-m', type=str, help='load from this checkpoint')
    args = parser.parse_args()

    dataset_path = '../../../Data/ABAW4'

    transform = image_test_saw2(img_size=224)
    img_path = dataset_path +  '/cropped_aligned/' + args.img
    print(img_path)
    img = pil_loader(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    net = MEFARG(num_classes=12, backbone=args.arc)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        img = img.cuda()

    print("Load form | {} ]".format(args.model))
    net = load_state_dict(net, args.model)
    net.eval()
    yhat_va, yhat_ex, yhat_au, _ = net(img)

    dataset = SAW2(dataset_path, train=False, transform=image_test_saw2(img_size=224))
    
    #index = [i for i, x in enumerate(dataset.data_list) if x[0] == args.img]
    img, va, ex, au, mask_va, mask_expr, mask_au = dataset[200]
    print('name: ', dataset.data_list[200][0])

    print('img: ', img)
    print('va : ', va)
    print('exp: ', ex)
    print('aus: ', au)

    print('yhat_va: ', yhat_va)
    print('yhat_ex: ', yhat_ex)
    print('yhat_au: ', yhat_au)

    i = torch.argmax(yhat_ex, dim=1)
    print('predict_va: ', yhat_va)
    print('predict_ex: ', i)
    print('predict_au: ', yhat_au >= 0.5)

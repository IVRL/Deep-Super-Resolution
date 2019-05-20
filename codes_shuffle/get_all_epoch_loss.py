#!/usr/bin/python
from math import log10
import cv2
import numpy as np
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from utils import *
from imageio import imread, imwrite
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import *
from dataset import Hdf5Dataset, FolderDataset
from utils import get_model_dir

# Training settings
parser = argparse.ArgumentParser(description="EI2019-SuperWavelets")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")

# Continuing training
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')

# Experiment configurations
parser.add_argument('--scale', default=2, type=int, help='Downscaling factor (default: 2)')
parser.add_argument('--wavelets', default=False, type=lambda s: s == 'True', help='Whether to use wavelets or not (default: False)')
parser.add_argument('--l-channel', default=False, type=lambda s: s == 'True', help='Whether to use the luminance or not (default: False)')
parser.add_argument('--dataset', default='valid', type=str, help='Dataset for testing')

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True

    print("===> Loading datasets")
    scale = opt.scale
    wavelets = opt.wavelets
    l_channel = opt.l_channel
    test_dataset = 'valid'
    
    valid_set = Hdf5Dataset(base='valid', scale=scale, wavelets=wavelets, l_channel=l_channel)
    validation_data_loader = DataLoader(dataset=valid_set, num_workers=1, batch_size=opt.batchSize)

    print("===> Building model")
    model = ResidualLearningNet(wavelets=wavelets, l_channel=l_channel)
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    model = model.cuda()
    criterion = criterion.cuda()
    
    # Loading previous models
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    minloss = 10000000000000;
    bestep = 1;
    print(get_model_dir(opt.scale, opt.wavelets, opt.l_channel))
    file = open("%s.txt"%get_model_dir(opt.scale,opt.wavelets, opt.l_channel),'a')
    for epoch in range(1, opt.nEpochs + 1): 
        model_dir = "checkpoint/%s" % get_model_dir(opt.scale, opt.wavelets, opt.l_channel)
        model_out_path = "%s/model_epoch_%d.pth" % (model_dir, epoch)
        state = {"epoch": epoch ,"model": model}
        checkpoint = torch.load(model_out_path)
        model.load_state_dict(checkpoint["model"].state_dict())
        loss = validate(validation_data_loader, model, criterion)
        if (loss < minloss):
            minloss = loss
            bestep = epoch
        psnr = 10 * log10(255 * 255 / loss)
        file.write('%f\n'%(psnr))

def validate(validation_data_loader, model, criterion):
    model.eval()
    total_loss = 0
    #print("===> Validation")
    for iteration, batch in enumerate(validation_data_loader, 1):
        data, target = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
        output = model(data)
        
        loss = criterion(output, target)
        total_loss += loss.item()
        
    count = len(validation_data_loader)
    #print("Validation Loss: {:.10f}".format(total_loss / count))
    return total_loss / count

if __name__ == "__main__":
    main()

#!/usr/bin/python
from skimage.measure import compare_ssim as ssim
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
import pandas as pd

# Training settings
parser = argparse.ArgumentParser(description="EI2019-SuperWavelets")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")

# Continuing training
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')

# Experiment configurations
parser.add_argument('--scale', default=2, type=int, help='Downscaling factor (default: 2)')
parser.add_argument('--wavelets', default=False, type=lambda s: s == 'True', help='Whether to use wavelets or not (default: False)')
parser.add_argument('--l-channel', default=False, type=lambda s: s == 'True', help='Whether to use the luminance or not (default: False)')
parser.add_argument('--y-channel', default=True, type=lambda s: s == 'True', help='Whether to use Y channel')
parser.add_argument('--dataset', default='valid', type=str, help='Dataset for testing')

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True

    print("===> Loading datasets")
    scale = opt.scale
    wavelets = opt.wavelets
    l_channel = opt.y_channel
    test_dataset = 'valid'
    
    valid_set = Hdf5Dataset(base='valid', scale=scale, wavelets=wavelets, l_channel=l_channel)
    validation_data_loader = DataLoader(dataset=valid_set, num_workers=1, batch_size=opt.batchSize)

    print("===> Building model")
    model = ResidualLearningNet(wavelets=wavelets, l_channel=l_channel)
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    model = model.cuda()
    criterion = criterion.cuda()
    
    minloss = 10000000000000;
    losses = []
    bestep = 1;
    print("===> Training")
    file = open("results/%s.txt" % get_model_dir(opt.scale, opt.wavelets, opt.l_channel, opt.y_channel), "w") 
    for epoch in range(1, opt.nEpochs + 1):
        model_dir = "checkpoint/%s" % get_model_dir(opt.scale, opt.wavelets, opt.l_channel, opt.y_channel)
        model_out_path = "%s/model_epoch_%d.pth" % (model_dir, epoch)
        state = {"epoch": epoch ,"model": model}
        checkpoint = torch.load(model_out_path)
        model.load_state_dict(checkpoint["model"].state_dict())
        loss = validate(validation_data_loader, model, criterion)
        losses.append(loss)
        if (loss < minloss):
            minloss = loss
            bestep = epoch
        psnr = 10 * log10(255 * 255 / loss)
        file.write("%.2f\n" % psnr)
    file.close()
    print("best epoch:%d, loss: %f" % (bestep, minloss))

    # Configuration
    dataset=opt.dataset
    scale=opt.scale
    wavelets=opt.wavelets
    l_channel=opt.l_channel
#     model_dir = "%s" % get_model_dir(opt.scale, opt.wavelets, opt.l_channel)
    # model_path = "%s/model_epoch_%d.pth" % (model_dir, epoch)
    

    model_dir = "%s" % get_model_dir(opt.scale, opt.wavelets, opt.l_channel, opt.y_channel)
    best_epoch = bestep
    model_out_path = "checkpoint/%s/model_epoch_%d.pth" % (model_dir, best_epoch)
    print("===> Loading Best Epoch %d from %s" % (best_epoch, model_out_path))
    checkpoint = torch.load(model_out_path)
    model.load_state_dict(checkpoint["model"].state_dict())

    # Configuration
    dataset=opt.dataset
    scale=opt.scale
    wavelets=opt.wavelets
    l_channel=opt.l_channel
    # model_path = "%s/model_epoch_%d.pth" % (model_dir, epoch)

    datasets = ["BSDS100", "Manga109", "Set14", "Set5", "Urban100"]

    flag = True;
    for i in range(0, 5):
        results = []
        dataset = datasets[i];
        
        print("===>Dataset %s" % dataset)
        # Load dataset
        folder_dataset = FolderDataset(dataset=dataset, scale=scale, wavelets=wavelets, l_channel=True)
        validation_data_loader = DataLoader(dataset=folder_dataset, num_workers=1, batch_size=1)

        # Build output directory based on configuration
        main_results_dir = "results/"
        current_results_dir = "%s/%s/%s" % (main_results_dir, dataset, model_dir)
        if not os.path.exists(current_results_dir):
            os.makedirs(current_results_dir)
        flag=True;
        for batch in validation_data_loader:
            file_name, data, target = batch
            # Predict super res
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            # Transform to numpy and save
            basename = os.path.basename(file_name[0])
            
            image = to_np(output)
            gd = to_np(target)
            if opt.wavelets:
                image = get_spatial(image)
                gd = get_spatial(gd)
            else:
                data = to_np(data)
                bic_ssim = ssim(data.astype('float'), gd.astype('float'),
                        multichannel=True, data_range=gd.max()-gd.min())
                
                mse = np.mean((data.astype('float') - gd.astype('float')) ** 2)
                bic_psnr = 10 * log10(255 * 255/mse)

            image = np.clip(image, 0, 255)

            
            pred_ssim = ssim( image.astype('float') ,gd.astype('float') ,
                    multichannel=True, data_range=gd.max()-gd.min())
            
            mse = np.mean( (image.astype('float') - gd.astype('float')) ** 2 )
            pred_psnr = 10 * log10(255 * 255 / mse)
            if opt.wavelets:
                results.append({'name': basename, 'pred_psnr':pred_psnr, 'pred_ssim': pred_ssim})
            else:
                results.append({'name': basename, 'bic_psnr': bic_psnr, 'bic_ssim': bic_ssim, 'pred_psnr':pred_psnr, 'pred_ssim': pred_ssim})
                
        df = pd.DataFrame(results)
        df.set_index('name')
        df.to_csv('%s/result.csv' % (current_results_dir), index=False)

def validate(validation_data_loader, model, criterion):
    model.eval()
    total_loss = 0
    #print("===> Validation")
    for iteration, batch in enumerate(validation_data_loader, 1):
        with torch.no_grad():
            data, target = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
            output = model(data)

            loss = criterion(output, target)
        total_loss += loss.item()
    count = len(validation_data_loader)
    #print("Validation Loss: {:.10f}".format(total_loss / count))
    return total_loss / count

if __name__ == "__main__":
    main()

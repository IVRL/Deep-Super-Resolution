from dataset import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import os, argparse
from imageio import imread, imwrite
from utils import to_np, get_model_dir, get_spatial

# SOME OUT OF MEMORY ISSUES for large images!!!
# @Ruofan do your cropping image into 4 trick :)

# Experiment configurations
parser = argparse.ArgumentParser(description="PIRM2018")
parser.add_argument('--epoch', default=99, type=int, help='Model epoch to use for testing')

parser.add_argument('--scale', default=2, type=int, help='Downscaling factor (default: 2)')
parser.add_argument('--wavelets', default=False, type=lambda s: s == 'True', help='Whether to use wavelets or not (default: False)')
parser.add_argument('--l-channel', default=False, type=lambda s: s == 'True', help='Whether to use the luminance or not (default: False)')
parser.add_argument('--dataset', default='valid', type=str, help='Dataset for testing')

opt = parser.parse_args()
print(opt)

# Configuration
dataset=opt.dataset
scale=opt.scale
wavelets=opt.wavelets
l_channel=opt.l_channel
model_dir = get_model_dir(scale, wavelets, l_channel)
model_path = 'checkpoint/%s/model_epoch_%d.pth' % (model_dir, opt.epoch)

# Load dataset
folder_dataset = FolderDataset(dataset=dataset, scale=scale, wavelets=wavelets, l_channel=l_channel)
validation_data_loader = DataLoader(dataset=folder_dataset, num_workers=1, batch_size=1)
# Load model
model = torch.load(model_path, map_location=lambda storage, loc: storage)["model"]
model.cuda().eval()

# Build output directory based on configuration
main_results_dir = "results/"
current_results_dir = "%s/%s/%s" % (main_results_dir, dataset, model_dir)
if not os.path.exists(current_results_dir):
    os.makedirs(current_results_dir)

for batch in validation_data_loader:
    file_name, data, target = batch
    # Predict super res
    data, target = Variable(data).cuda(), Variable(target).cuda()
    output = model(data)
    
    # Transform to numpy and save
    basename = os.path.basename(file_name[0])
    image = to_np(output)
    
    if opt.wavelets:
        image = get_spatial(image)
        
    image = image.astype(np.uint8)
    imwrite('%s/%s' % (current_results_dir, basename), image)
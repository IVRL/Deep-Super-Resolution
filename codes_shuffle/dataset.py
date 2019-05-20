import h5py
import torch.utils.data as data
import torch
from torchvision import transforms
from functools import partial
import numpy as np
from imageio import imread
import glob
from utils import get_wavelet

class Hdf5Dataset(data.Dataset):
    def __init__(self, base='train', scale=2, wavelets=False, l_channel=False):
        super(Hdf5Dataset, self).__init__()
        
        base = base + '/'
        if l_channel:
            base = base + 'l'
        
        self.wavelets = wavelets
        
        self.hr_dataset = h5py.File(base + 'hr.h5')['/data']
        self.lr_dataset = h5py.File(base + 'lr%d.h5' % scale)['/data']

    def __getitem__(self, index):
        x, y = self.hr_dataset[index], self.lr_dataset[index]
        if self.wavelets:
            x, y = get_wavelet(x), get_wavelet(y)
        return y, x

    def __len__(self):
        return self.hr_dataset.shape[0]
    
class FolderDataset(data.Dataset):
    def __init__(self, dataset='valid', scale=2, wavelets=False, l_channel=False):
        super(FolderDataset, self).__init__()
        
        base = '/scratch/mfr/GeneratedSetsY/%s/' % dataset
        if l_channel:
            base = base + 'l'
        
        self.wavelets = wavelets
        
        self.hr_files = sorted(glob.glob(base + 'hr/*.png'))
        self.lr_files = sorted(glob.glob(base + 'lr%d/*.png' % scale))
    
    def __getitem__(self, index):
        x = np.atleast_3d(imread(self.hr_files[index]))
        x = np.rollaxis(x, 2, 0)
        
        y = np.atleast_3d(imread(self.lr_files[index]))
        y = np.rollaxis(y, 2, 0)
        
        if self.wavelets:
            x, y = get_wavelet(x), get_wavelet(y)
        return self.hr_files[index], torch.from_numpy(y).float(), torch.from_numpy(x).float()
        
    def __len__(self):
        return len(self.hr_files)


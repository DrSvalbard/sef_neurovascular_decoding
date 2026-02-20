import torch
import torch.nn as nn 
from torch.utils.data import Dataset

import numpy as np 
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d

# Creating the dataset class
class FUS_LFP_Dataset(Dataset):
    '''
    INIT
    '''
    def __init__(self,
                f_us,
                lfp_dict,
                indices,
                temporal_shift = 10,
                random_noise = 0.01,
                fus_smooth = 5,
                lfp_smooth = 1,
                split='train',
                window_past=150,
                window_future=400,
                target_len=200,
                target_pretrig = 10
                ):
        
        '''
        :param self: 
        :param f_us: PCA-ed or raw_fus DATA (Time, Channel)
        :param lfp_dict: {'Alpha','Beta','Gamma'} dict (Time,)
        :param indices: Extrated indices for the dataset
        :param temporal_shift: Temporal randomisation (1 = 10ms)
        :param random_noise: White noice (0<noise<1) added to f-US data
        :param fus_smooth: 1D smoothing (1 = 10ms)
        :param lfp_smooth: 1D smoothing (1 = 10ms)
        :param split: 'train', 'val'
        :param window_past: Before the idx (1 = 10ms)
        :param window_future: After the idx (1 = 10ms)
        :param target_len: LFP length kept (1 = 10ms)
        '''

        self.f_us = f_us  # [Time, PCx] 
        self.split = split
        # Stacking to have [Time, 3]
        self.lfp = np.stack([lfp_dict['Alpha'], 
                             lfp_dict['Beta'], 
                             lfp_dict['Gamma'],
                             lfp_dict['HGamma']], axis=1).squeeze()
        
        # trying to predict only one ?
        self.lfp = lfp_dict['HGamma']
        if self.lfp.ndim == 1:
            self.lfp = self.lfp[:, np.newaxis]
        # end 

        self.indices = indices 
        self.window_past = window_past
        self.window_future = window_future
        self.target_len = target_len
        self.temporal_shift = temporal_shift
        self.noise_value = random_noise
        self.fus_smooth = fus_smooth
        self.lfp_smooth = lfp_smooth
        self.target_pretrig = target_pretrig
    '''
    LEN
    '''
    def __len__(self):
        return len(self.indices)
    
    '''
    GET ITEM
    '''
    def __getitem__(self, idx):
        t = self.indices[idx]

        # fUS input : slice [t-win_past à t+wn_future] -> [Time, PCx]
        # .T : [PCx, Time]
        len_sig = self.window_future + self.window_past
        x = self.f_us[t - self.window_past : t + self.window_future, :].copy()
        x = uniform_filter1d(x, size=self.fus_smooth, axis=0)
        
        # Trigger channel
        trigger_channel = np.zeros((x.shape[0], 1))
        trigger_channel[self.window_past] = 1.0

        x_final = np.concatenate([x, trigger_channel], axis=1)
        x_time = torch.from_numpy(x_final.T).float()

        if x_time.shape[1] < len_sig-self.window_past:
            diff = len_sig - x_time.shape[1]
            x_time = nn.functional.pad(x_time, (0, diff)) # On rajoute des 0 à la fin du temps

        # Target : slice [t à t+100] -> [100, 3]
        # .T :[3, 100] 
        y = self.lfp[t - self.target_pretrig : t + self.target_len, :].T
        y = uniform_filter1d(y, size=self.lfp_smooth, axis=1)
        # normalize
        # y = (y - y.mean(axis=1, keepdims=True)) / (y.std(axis=1, keepdims=True) + 1e-6)

        return x_time.permute(1, 0), torch.from_numpy(y).float()
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import gc
# custom function
from utils.classes.dataset_manager import FUS_LFP_Dataset

# ----------------------------- LOAD -----------------------------

def load_data(path_npz, fs=100):
    '''
    :param path_npz: str
    :param fs: int (fs=100)
    '''
    data = np.load(path_npz, allow_pickle=True)
    # load
    fus_data = data['fus'].astype(np.float32).T # [time_pts, 700]
    lfp_alpha = data['alpha'][:, np.newaxis].astype(np.float32)   # [time_pts, 1]
    lfp_beta = data['beta'][:, np.newaxis].astype(np.float32)    # [time_pts, 1]
    lfp_gamma = data['gamma'][:, np.newaxis].astype(np.float32)  # [time_pts, 1]
    lfp_hgamma = data['hgamma'][:, np.newaxis].astype(np.float32)
    events = data['events']
    # clear a bit of the ram
    del data
    gc.collect()
    #  
    lfp_dict = {
    'Alpha': lfp_alpha,
    'Beta' : lfp_beta,
    'Gamma': lfp_gamma,
    'HGamma': lfp_hgamma
    }   

    return fus_data, lfp_dict, events

# ----------------------------- PCA -----------------------------

def data_pca(train_set, fus_data, n_pca):
    '''
    :param fus_data: np.array [Time, n_roi]
    :param n_pca: int (e.g. 15)
    '''
    # create the pca object
    pca = PCA(n_components=n_pca)
    # pca
    pca.fit(train_set)
    components = pca.transform(fus_data)
    explained_var = pca.explained_variance_ratio_
    total_var = np.sum(explained_var) * 100
    print(f'Explained variance using {n_pca} components : {total_var:.1f}%')

    return components

# ----------------------------- DATA NORMALIZATION -----------------------------

def z_score(data_in):
    n_mean = np.mean(data_in, axis=0)
    n_std = np.std(data_in, axis=0) + 1e-6
    return (data_in-np.mean(data_in, axis=0))/(np.std(data_in, axis=0) + 1e-6), n_mean, n_std

def data_normalization(trainset_fus, trainset_lfp, fus_pca, lfp_dict):
    _, train_mean, train_std = z_score(trainset_fus) # -> train_std cannot be 0 (+1e-6 as backup in z_score code)
    fus_norm = (fus_pca-train_mean)/train_std

    for idx in lfp_dict:
        _, lfp_mean, lfp_std = z_score(trainset_lfp[idx]) # -> lfp_std cannot be 0 (+1e-6 as backup in z_score code)
        lfp_dict[idx] = (lfp_dict[idx]-lfp_mean)/lfp_std

    return fus_norm, lfp_dict

# ---------------------------- IDX EVENTS -----------------------------

def extract_idx_events(events, len_fus, idx_fixation:int=230, shift:int=1, split_proportion=0.8, fs=100, window_size=500):
    '''
    :param events: np.array
    :param idx_fixation: int (Claron et al. task : 230)
    :param shift: int (Claron et al. task : 1)
    :param split_proportion: float 0<sp<1
    :param fs: float (100)
    '''

    # Extract triggers
    ev = events[:,1]
    ev_roll = np.roll(ev,-1)
    truth_event = np.logical_and(ev==idx_fixation, ev_roll != idx_fixation)
    trigger_times = events[np.where(truth_event)[0][1:-2]+shift,0] 
    start_time = events[np.where(events[:,1]==idx_fixation)[0][0],0] 
    trigger_times = (trigger_times-start_time) * fs
    trigger_times = np.round(trigger_times).astype(int)
    #
    max_idx = len_fus - window_size
    trigger_times = trigger_times[trigger_times < max_idx]

    # randomise and split
    # np.random.shuffle(trigger_times) # randomize order
    split = int(split_proportion * len(trigger_times))
    train_idx = trigger_times[:split]
    test_idx = trigger_times[split:]
    print(f"Size : {len(train_idx)} train, {len(test_idx)} test")

    return train_idx, test_idx


# ----------------------------- TORCH READY DATAS -----------------------------

def torch_ready_dataset(fus_in, lfp_dict, idx, batch_size, window_past=50, window_future=300 ,split='Train'):
    '''
    :param fus_in: pca-ed fus
    :param lfp_dict: dict
    :param idx: from extract_idx_events
    :param batch_size: int (16)
    :param split: str 'Train' / 'Val'
    '''
    dataset = FUS_LFP_Dataset(fus_in, lfp_dict, idx, window_past=window_past, window_future=window_future, target_len=200)
    if split == 'Train':
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    return loader


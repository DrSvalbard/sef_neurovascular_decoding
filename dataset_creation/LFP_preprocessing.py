#%%
from neo.io import PlexonIO
import matplotlib.pyplot as plt
import scipy.signal as sp_sig
from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, decimate
from scipy.ndimage import uniform_filter1d
import sys
import gc
#%% 
path_plx = '/Users/julienclaron/Desktop/InverseProblem/BIQ_HF_FUS/process_2506/Se25062020.plx'
reader = PlexonIO(filename=path_plx)
bl = reader.read_block()

# %%
seg = bl.segments[0].analogsignals
LFP_general = seg[0]
LFP_signal = np.squeeze(LFP_general[:,0].magnitude).astype(np.float32)

del seg, LFP_general
gc.collect()

# %% 

def extract_idx_events(events, len_fus, idx_fixation:int=230, shift:int=1, split_proportion=0.95, fs=1000, window_size=3500):
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

# %%
# On cherche l'event qui contient les codes Strobed (souvent nommé 'Strobed' ou '257')
strobed_event = None
for ev in bl.segments[0].events:
    if 'Strobed' in ev.name or '257' in ev.name:
        strobed_event = ev
        break

# Les temps des évènements (en secondes)
event_times = strobed_event.times.magnitude 

# Les codes envoyés (les "labels" ou "values")
event_codes = strobed_event.labels

# On crée une matrice propre [Temps, Code]
events_matrix = np.column_stack((event_times, event_codes.astype(int)))
print("Event matrix OK !")

# %% 

idx_to_test, _ = extract_idx_events(events_matrix, len(LFP_signal))
# %%

def z_scr(input):
    return (input-np.mean(input[:100]))/(np.std(input[:100])+1e-6)

start_idx_lfp = np.where(events_matrix == 230)[0][0]
start_idx_lfp = np.round(events_matrix[start_idx_lfp,0]*100).astype(int)
print(start_idx_lfp)
# %%

def get_envelope(signal, low, high, fs_original, fs_target):
    nyq = 0.5 * fs_original
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    
    # 1. Filtrage Band-pass (Beta)
    filtered = filtfilt(b, a, signal)
    
    # 2. Hilbert pour l'enveloppe brute
    envelope = np.abs(hilbert(filtered))
    
    # 3. Resampling immédiat (pour passer à 100Hz)
    # Decimate applique un filtre passe-bas pour éviter l'aliasing
    q = int(fs_original / fs_target)
    envelope_resampled = decimate(envelope, q, ftype='fir')
    
    # 4. Lissage léger au format cible (100Hz)
    # size=10 à 100Hz = 100ms de lissage. C'est suffisant.
    smooth_e = uniform_filter1d(envelope_resampled, size=10, mode='reflect')
    
    return smooth_e

#%% 
bands = {
    'Alpha': (8, 12),
    'Beta': (13, 30),
    'Gamma': (30, 80),
    'High-Gamma': (80, 200)
}

lfp_envelopes = {}
fs_lfp = 1000  # À vérifier selon tes réglages Plexon
fs_fus = 100   # Ta fréquence fUS

for name, (low, high) in bands.items():
    print(f"Traitement de la bande {name}...")
    lfp_envelopes[name] = get_envelope(np.squeeze(LFP_signal), low, high, fs_lfp, fs_fus)

# z-scoring
for name in lfp_envelopes:
    env = lfp_envelopes[name]
    lfp_envelopes[name] = (env - np.mean(env)) / np.std(env).astype(np.float32)



#%%

np.savez('lfp_envelope_z_scrd.npz',
        hgamma = lfp_envelopes['High-Gamma'][start_idx_lfp:].astype(np.float32),
        gamma = lfp_envelopes['Gamma'][start_idx_lfp:].astype(np.float32),
        beta = lfp_envelopes['Beta'][start_idx_lfp:].astype(np.float32),
        alpha = lfp_envelopes['Alpha'][start_idx_lfp:].astype(np.float32))
# %%

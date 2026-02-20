#%%
import numpy as np
from scipy.signal import detrend
#%% 
old_fus = np.load('/Users/julienclaron/Desktop/InverseProblem/TransUNET_PCA/datas_npz/new_data_TransUnet_Se20200625.npz')
new_lfp = np.load('/Users/julienclaron/Desktop/InverseProblem/TransUNET_PCA/datas_npz/lfp_envelope_z_scrd.npz')

#%% 

np.savez('data_preprocess_Se20200625_20260216.npz',
                    fus = old_fus['fus'].astype(np.float32),
                    hgamma = detrend(new_lfp['hgamma'].astype(np.float32)),
                    gamma = detrend(new_lfp['gamma'].astype(np.float32)),
                    beta = detrend(new_lfp['beta'].astype(np.float32)),
                    alpha = detrend(new_lfp['alpha'].astype(np.float32)),
                    events = old_fus['events'],
                    )
# %%

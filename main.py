# %% --- autoreload jupyter ---
%load_ext autoreload
%autoreload 2


# %% --- Changing the dir ---
import sys, os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'     # -> No fallback
# Add to path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# %% --- Imports --- 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.ndimage import uniform_filter1d

# custom import
from utils import pearson_loss , TemporalEnsembleLoss
from utils import load_data, data_pca, data_normalization, extract_idx_events, torch_ready_dataset
from utils import master_network
from utils import get_loss_weights
from utils import batch_time_mask

# plot
from IPython.display import clear_output
import matplotlib.pyplot as plt

# %% --- path npz ---
path_npz = 'data/data_20260216.npz'
fus_data, lfp_dict, events = load_data(path_npz)

# %% --- Data preprocessing ---
# super pixels
# Note: Global Z-score normalization applied per channel. 
# Biological purpose: equalize variance between major blood vessels and capillaries.
# ML note: Assumes session-level stationarity. For strict causal deployment, a rolling-window Z-score should be implemented to prevent temporal leakage.
fus_data_norm = (fus_data - np.mean(fus_data,axis=0))/(np.std(fus_data,axis=0)+1e-6)
T, P = fus_data_norm.shape
reshaped_fus_data = fus_data_norm[:,:700].reshape(T, 100, 7)
reshaped_fus_data = np.mean(reshaped_fus_data,axis=2)
# indexes
train_idx, test_idx = extract_idx_events(events, reshaped_fus_data.shape[0])

max_train_idx, min_test_idx = np.max(train_idx), np.min(test_idx)
# verify max < min
if min_test_idx < max_train_idx:
    raise ValueError('Min test index < Max train index')

train_fus = reshaped_fus_data[:max_train_idx,:] # -> [T, SuperPixel]
# pca
pca_fus = data_pca(train_fus, reshaped_fus_data, 24) # PCA on train_fus, applied to reshaped_fus_data, to avoid temporal leakage
# normalisation
train_lfp = dict()
for idx in lfp_dict:
    train_lfp[idx] = lfp_dict[idx][:max_train_idx]

pca_fus_norm, lfp_dict_norm = data_normalization(train_fus, train_lfp ,pca_fus, lfp_dict)

# train_dl 
cut_pca_fus = pca_fus_norm[:,:]

window_past, window_future = 10, 410
train_dl = torch_ready_dataset(cut_pca_fus, lfp_dict_norm, train_idx, 32, split='Train', window_past=window_past, window_future=window_future)
# val_dl 
test_dl = torch_ready_dataset(cut_pca_fus, lfp_dict_norm, test_idx, 32, split='Test', window_past=window_past, window_future=window_future)

# %% --- Model misc ---

n_epochs = 50
train_losses = []
best_loss = float(1e10) # init infinite loss
total_steps = len(train_dl) * n_epochs
# mps or cpu
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
mse_loss = nn.MSELoss()
hybrid_loss = TemporalEnsembleLoss().to(device)
jitter = 10
noise_lvl = 0.05
w_len = window_past + window_future

# %% --- Init the model --- 

model = master_network(num_pcs=cut_pca_fus.shape[1]+1, # -> +1 for the trigger channel
                       embed_dim=64,
                       n_head=4,
                       num_layers=2,
                       max_len=w_len,
                       output_len=210)

model.to(device)
# Optimizer (AdamW -> Good with transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
# Scheduler and warm-up
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr = 1e-3, steps_per_epoch=total_steps, epochs=n_epochs, pct_start=0.1)
# Xavier init
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # Xavier init
        nn.init.xavier_normal_(m.weight.data, gain=1.0) 
# Apply the weight init
model.apply(weights_init)
print(f' Used device : {device}')

# %% --- Batch shape verification ---
batch_x, batch_y = next(iter(train_dl))
print('--- Verification ---')
print(f'Batch fus (x) : {batch_x.shape} [Should be [Batch, Time, Feature]]')
print(f'Batch LFP (y) : {batch_y.shape} [Should be [Batch, Feature, Time]]')

# %% --- Loop --- 
# metric trackers
history = {
    'train_mse': [], 'train_pearson': [],
    'val_mse': [], 'val_pearson': []
}

# Loop
for epoch in range(n_epochs):
    # Put the model in training mode
    model.train()
    # metrics
    epoch_mse_sum = 0.0
    epoch_pearson_sum = 0.0
    n_train_batches = 0
    # tqdm progress bar over the train dataset
    pbar = tqdm(train_dl)

    for i,(x_batch, y_batch) in enumerate(pbar):
        # Jittering (data augmentation)
        start = np.random.randint(0, jitter*2 + 1)
        x_batch = x_batch[:, start : start + w_len - jitter*2, :]
        x_batch = batch_time_mask(x_batch)
        # to MPS
        x_batch = x_batch.to(device)  # -> on GPU
        y_batch = y_batch.to(device)  # -> on GPU

        # Cleaning the model
        optimizer.zero_grad()

        # model(x)
        x_batch = x_batch.transpose(1,2).contiguous()
        output = model(x_batch)

        # Loss
        w_mse, w_pearson = get_loss_weights(epoch)
        mse_loss_val = mse_loss(output, y_batch)
        pearson_loss_val = pearson_loss(output, y_batch)
        #loss = w_mse*mse_loss_val + w_pearson*pearson_loss_val
        loss = hybrid_loss(output, y_batch)

        # Gradian back-propagation
        loss.backward()
        optimizer.step()

        # tqdm update
        pbar.set_description(f"Epoch {epoch+1}")
        pbar.set_postfix(loss=f'{loss.item():.4f}')

        # loss storage for post-analysis and plotting
        with torch.no_grad():
            epoch_mse_sum += mse_loss_val
            epoch_pearson_sum += pearson_loss_val
            n_train_batches += 1

            train_losses.append(loss.detach().cpu().item())

    history['train_mse'].append(epoch_mse_sum.detach().cpu().item()/n_train_batches)
    history['train_pearson'].append(epoch_pearson_sum.detach().cpu().item()/n_train_batches)
    
    # Update the scheduler
    scheduler.step()
    # Eval
    model.eval()
    val_mse_sum = 0.0
    val_pearson_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test = x_test[:,jitter:w_len-jitter,:]

            x_test = x_test.to(device) # -> to MPS
            y_test = y_test.to(device) # -> to MPS
            # model(x)
            x_test = x_test.transpose(1,2) # -> [Batch, Feature, Time]
            output = model(x_test)
            # values
            mse_loss_val_test = mse_loss(output, y_test)
            pearson_loss_val_test = pearson_loss(output, y_test)
            # Store
            val_mse_sum += mse_loss_val_test.detach().cpu().item()
            val_pearson_sum += pearson_loss_val_test.detach().cpu().item()
            num_batches += 1


        history['val_mse'].append(val_mse_sum/num_batches)
        history['val_pearson'].append(val_pearson_sum/num_batches)

        if val_pearson_sum/num_batches < best_loss:
            best_loss = val_pearson_sum/num_batches
            torch.save(model.state_dict(), 'models/best_model_pearson.pth')
            tqdm.write(f'New model saved (Pearson : {(1-best_loss):.4f}) @ epoch : {epoch}')



    clear_output(wait=True) # del the previous graph
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()
    # Graphique MSE
    ax1.plot(history['train_mse'], color='orange')
    ax1.set_title("Convergence de l'Amplitude (MSE)")

    # Graphique Pearson
    ax2.plot(history['train_pearson'], color='green')
    ax2.set_title('Convergence de la Forme (Pearson)')

    # Graphique Pearson
    ax3.plot(history['val_mse'], color='firebrick')
    ax3.set_title("Convergence de la l'Amplitude (MSE) sur le test")

    # Graphique Pearson
    ax4.plot(history['val_pearson'], color='darkolivegreen')
    ax4.set_title('Convergence de la Forme (Pearson) sur le test')

    plt.show()      





# %% Show
model.eval()
with torch.no_grad():
    # Prend un batch de test
    x_test, y_test = next(iter(test_dl))
    x_test = x_test[:,jitter:w_len-jitter,:]
    x_test = x_test.transpose(1,2)
    pred = model(x_test.to(device)).cpu()

# Visualise le premier exemple
plt.figure(figsize=(12, 4))
plt.plot(y_test[0, 0, :], label='Beta Réel', alpha=0.7)
plt.plot(pred[0, 0, :], label='Beta Prédit (Transformer)', color='red')
plt.title(f"Vérification sur le Test Set - Correlation: {np.corrcoef(y_test[0,0,:], pred[0,0,:])[0,1]:.2f}")
plt.legend()
plt.show()

# %% Attention

def visualize_attention(model, x_sample,jitter = jitter, w_len = w_len, device='mps'):
    model.eval()
    # x_sample shape: [1, PCx, 350]
    x_sample = x_sample[:,jitter:w_len-jitter,:].transpose(2, 1)
    x_sample = x_sample.to(device)
    
    # 1. Manually pass through Encoder
    tokens = model.encoder(x_sample) # [1, 64, 350]
    tokens = tokens.transpose(1, 2)  # [1, 350, 64]
    tokens = model.pe(tokens)
    
    # 2. Extract attention from Transformer layers
    # We use the internal layers of model.transformer.transformer
    attn_maps = []
    
    # Standard PyTorch TransformerEncoder stores layers in .layers
    for layer in model.transformer.transformer.layers:
        # Get attention weights: (Target_Len, Source_Len)
        # Note: We need to set average_attn_weights=True
        _, weights = layer.self_attn(tokens, tokens, tokens, average_attn_weights=True)
        attn_maps.append(weights.detach().cpu().numpy().squeeze())
        # Pass tokens to next layer to keep the flow
        tokens = layer(tokens)

    # 3. Plotting
    fig, axes = plt.subplots(1, len(attn_maps), figsize=(15, 5))
    for i, map in enumerate(attn_maps):
        im = axes[i].imshow(map, cmap='viridis', aspect='auto')
        axes[i].set_title(f'Layer {i+1} Attention')
        axes[i].set_xlabel('Source Time (fUS)')
        axes[i].set_ylabel('Target Time (LFP Tokens)')
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

# Usage:
x_test, _ = next(iter(test_dl))
visualize_attention(model, x_test[0:1])




# %% Cross-attention

def plot_cross_attention(model, x_sample, device='mps'):
    model.eval()
    with torch.no_grad():
        x_sample = x_sample.to(device)
        x_sample = x_sample[:,jitter:w_len-jitter,:].transpose(2, 1)
        x_sample = x_sample.to(device)
        
        # 1. Passage dans l'Encoder + Transformer Encoder
        enc_out = model.encoder(x_sample)
        enc_out = enc_out.transpose(1, 2)
        enc_out = model.pe(enc_out)
        memory = model.transformer(enc_out)
        
        # 2. Extraction des poids de la Cross-Attention
        B = memory.shape[0]
        queries = model.query_pos.expand(B, -1, -1)
        
        # On demande explicitement les poids (attn_output_weights)
        _, weights = model.decoder_cross_attn(queries, memory, memory, average_attn_weights=True)
        weights = weights.cpu().numpy()[0] # [100, 350]

    plt.figure(figsize=(12, 6))
    plt.imshow(weights, aspect='auto', cmap='magma')
    plt.title("Carte de la Cross-Attention (LFP vs fUS)")
    plt.ylabel("Points LFP (0 à 100)")
    plt.xlabel("Points fUS (0 à 350)")
    plt.colorbar(label="Poids d'attention")
    plt.show()

x_test, _ = next(iter(test_dl))
plot_cross_attention(model, x_test[0:1])

# %%

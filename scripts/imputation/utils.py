
import h5py
import numpy as np
import os
import pandas as pd

import rnn_cells

def _build_recurrent_cell(hidden_dim, dropout_keep_prob):
    return rnn_cells.LayerNormLSTMCell(
        hidden_dim, 
        use_recurrent_dropout=True,
        dropout_keep_prob=dropout_keep_prob
    )

def compute_n_batches(n_samples, batch_size):
    n_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches

def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return lengths

def maybe_mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)

def load_ngsim_trajectory_data(
        filepath, 
        traj_key=None,
        feature_keys=[
            'velocity',
            'relative_offset',
            'relative_heading',
            'length',
            'width',
            'lane_curvature',
            'markerdist_left',
            'markerdist_right',
            'accel',
            'jerk',
            'turn_rate_frenet',
            'angular_rate_frenet'
        ],
        target_keys=[
            'lidar_1'
        ],
        binedges=[10,15,25,50],
        max_censor_ratio=.3,
        max_len=None,
        train_ratio=.9,
        max_samples=None,
        shuffle=True,
        normalize=True,
        censor=50.):
    # select from the different roadways
    infile = h5py.File(filepath, 'r')
    if traj_key is None:
        x = np.vstack([infile[k].value for k in infile.keys()])
    else:
        x = np.copy(infile[traj_key].value)

    # enforce max_len
    if max_len is not None:
        x = x[:,:max_len,:]
    
    # pandas format for feature-based selection
    panel = pd.Panel(
        data=x, 
        minor_axis=infile.attrs['feature_names']
    )
    
    lengths = np.array(compute_lengths(panel[:,:,feature_keys]))
    n_samples, n_timesteps, input_dim = panel[:,:,feature_keys].shape

    # only a single target key implemented for now
    assert len(target_keys) == 1
    k = target_keys[0]
        
    # remove samples with too many censored values
    valid_sample_idxs = []
    for j, l in enumerate(lengths):
        invalid_idxs = np.where(panel[j,:l,k] == censor)[0]
        if len(invalid_idxs) / l < max_censor_ratio:
            valid_sample_idxs.append(j)
    valid_sample_idxs = np.array(valid_sample_idxs)
            
    # debugging size
    if max_samples is not None:
        valid_sample_idxs = valid_sample_idxs[:max_samples]
    
    # shuffle
    if shuffle:
        permute_idxs = np.random.permutation(len(valid_sample_idxs))
        valid_sample_idxs = valid_sample_idxs[permute_idxs]

    y = np.zeros((len(valid_sample_idxs), n_timesteps), dtype=int)
    lengths = lengths[valid_sample_idxs]
    x = np.array(panel[valid_sample_idxs,:,feature_keys])
    
    # discretize the targets
    y[:,:] = np.digitize(
        panel[valid_sample_idxs,:,k].T, 
        binedges, 
        right=True
    )
        
    # normalize features
    if normalize:
        x -= np.mean(x, axis=(1,2), keepdims=True)
        x /= np.std(x, axis=(1,2), keepdims=True) + 1e-8
    
    # train / val split
    train_idx = int(len(valid_sample_idxs) * train_ratio)
    train_x = x[:train_idx]
    train_y = y[:train_idx]
    train_lengths = lengths[:train_idx]
    val_x = x[train_idx:]
    val_y = y[train_idx:]
    val_lengths = lengths[train_idx:]
    
    data = dict(
        train_x=train_x,
        train_y=train_y,
        train_lengths=train_lengths,
        val_x=val_x,
        val_y=val_y,
        val_lengths=val_lengths,
        feature_names=feature_keys,
        target_names=target_keys,
    )
    
    return data


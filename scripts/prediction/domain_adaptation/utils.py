
import h5py
import numpy as np
import os
import sys

def maybe_mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)

def compute_n_batches(n_samples, batch_size):
    n_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches

def compute_batch_idxs(start, batch_size, size):
    if start >= size:
        return list(np.random.randint(low=0, high=size, size=batch_size))
    
    end = start + batch_size

    if end <= size:
        return list(range(start, end))

    else:
        base_idxs = list(range(start, size))
        remainder = end - size
        idxs = list(np.random.randint(low=0, high=size, size=remainder))
        return base_idxs + idxs

def evaluate(model, dataset):
    src_info = model.evaluate(dataset.xs, dataset.ys, np.zeros(len(dataset.xs)))
    tgt_info = model.evaluate(dataset.xt, dataset.yt, np.ones(len(dataset.xt)))
    return dict(src_info=src_info, tgt_info=tgt_info)

def report(train_info, val_info):
    print('\n---------------')
    print('train src loss: {:.4f}'.format(train_info['src_info']['task_loss']))
    print('train tgt loss: {:.4f}'.format(train_info['tgt_info']['task_loss']))
    print('val src loss: {:.4f}'.format(val_info['src_info']['task_loss']))
    print('val tgt loss: {:.4f}'.format(val_info['tgt_info']['task_loss']))
    print('----------------')

def build_datasets(data, batch_size):
    # import this here because some functions about are needed 
    # by the dataset and it throws an error when importing as usual
    from domain_adaptation_dataset import DomainAdaptationDataset
    dataset = DomainAdaptationDataset(
            data['src_x_train'],
            data['src_y_train'],
            data['tgt_x_train'],
            data['tgt_y_train'],
            batch_size=batch_size
        )
    val_dataset = DomainAdaptationDataset(
            data['src_x_val'],
            data['src_y_val'],
            data['tgt_x_val'],
            data['tgt_y_val'],
            batch_size=batch_size
        )
    return dataset, val_dataset

def normalize_combined(src_train, tgt_train, src_val, tgt_val):
    # count samples for weighting the respective means
    n_src = len(src_train)
    n_tgt = len(tgt_train)
    n = n_src + n_tgt
    src_ratio = n_src / n
    tgt_ratio = n_tgt / n

    # compute mean by weighting the source and target means by their sizes
    src_mean = np.mean(src_train, axis=0, keepdims=True)
    tgt_mean = np.mean(tgt_train, axis=0, keepdims=True)
    mean = src_ratio * src_mean + tgt_ratio * tgt_mean

    # center the features and compute std
    src_train -= mean
    tgt_train -= mean
    src_std = np.std(src_train, axis=0, keepdims=True) 
    tgt_std = np.std(tgt_train, axis=0, keepdims=True)
    std = (src_std * src_ratio + tgt_std * tgt_ratio) + 1e-8

    # finish normalizing train, and normalize val
    src_train /= std
    tgt_train /= std
    src_val = (src_val - mean) / std
    tgt_val = (tgt_val - mean) / std

    return src_train, tgt_train, src_val, tgt_val, mean, std

def normalize_individual(x_train, x_val):
    mean = np.mean(x_train, axis=0, keepdims=True)
    x_train -= mean
    std = np.std(x_train, axis=0, keepdims=True) + 1e-8
    x_train /= std

    x_val = (x_val - mean) / std
    return x_train, x_val, mean, std

def normalize(src_train, tgt_train, src_val, tgt_val, mode):
    if mode == 'combined':
        src_train, tgt_train, src_val, tgt_val, mean, std = normalize_combined(
            src_train, tgt_train, src_val, tgt_val)
        return src_train, tgt_train, src_val, tgt_val, dict(mean=mean, std=std)
    elif mode == 'individual':
        src_train, src_val, src_mean, src_std = normalize_individual(src_train, src_val)
        tgt_train, tgt_val, tgt_mean, tgt_std = normalize_individual(tgt_train, tgt_val)
        stats = dict(src_mean=src_mean, src_std=src_std, tgt_mean=tgt_mean, tgt_std=tgt_std)
        return src_train, tgt_train, src_val, tgt_val, stats

def to_multiclass(y):
    ret = np.zeros((len(y), 2))
    ret[:,0] = 1-y
    ret[:,1] = y
    return ret

def transform_frustratingly(src, tgt, names):
    n_samples, input_dim = src.shape
    rtn_src = np.zeros((n_samples, input_dim * 3))
    rtn_src[:,:input_dim] = src
    rtn_src[:,input_dim: 2 * input_dim] = src

    n_samples, input_dim = tgt.shape
    rtn_tgt = np.zeros((n_samples, input_dim * 3))
    rtn_tgt[:,:input_dim] = tgt
    rtn_tgt[:,input_dim*2: 3*input_dim] = tgt

    rtn_names = list(names)
    for n in names:
        rtn_names += [n + '_src']
    for n in names:
        rtn_names += [n + '_tgt']

    return rtn_src, rtn_tgt, rtn_names

def load_data(
        source_filepath, 
        target_filepath,
        max_src_train_samples=None,
        max_tgt_train_samples=None,
        debug_size=None,
        target_idx=4,
        train_split=.8,
        timestep=-1,
        mode=''):
    
    # load files and feature names and target names
    src_file = h5py.File(source_filepath, 'r')
    src_feature_names = src_file['risk'].attrs['feature_names']
    tgt_file = h5py.File(target_filepath, 'r')
    tgt_feature_names = tgt_file['risk'].attrs['feature_names']
    src_target_names = src_file['risk'].attrs['target_names'][target_idx]
    tgt_target_names = tgt_file['risk'].attrs['target_names'][target_idx]
    assert src_target_names == tgt_target_names

    # subselect src features to only those features also in the target set
    # need to do this in a manner such that the features align
    # accomplish this by iterating the target feature names
    # finding the src feature name that matches the current target
    # and add its index if it exists
    keep_idxs = []
    for tgt_name in tgt_feature_names:
        for i, src_name in enumerate(src_feature_names):
            if src_name == tgt_name:
                keep_idxs.append(i)
                break

    # check that the order of the features is the same
    assert all(src_feature_names[keep_idxs] == tgt_feature_names)
    src_feature_names = src_feature_names[keep_idxs]

    # select features and targets 
    # perform some subselection for debugging case
    if debug_size is not None:
        src_features = src_file['risk/features'][:debug_size]
        src_features = src_features[...,keep_idxs]
        src_targets = src_file['risk/targets'][:debug_size,target_idx]
        tgt_features = tgt_file['risk/features'].value[:debug_size]
        tgt_targets = tgt_file['risk/targets'][:debug_size,target_idx]
    else:
        src_features = src_file['risk/features'].value[...,keep_idxs]
        src_targets = src_file['risk/targets'][:,target_idx]
        tgt_features = tgt_file['risk/features'].value
        tgt_targets = tgt_file['risk/targets'][:,target_idx]

    # subselect the last timestep for now
    if len(src_features.shape) > 2:
        src_features = src_features[:,timestep]
        tgt_features = tgt_features[:,timestep]

    # apply mode-specific transforms
    if mode == 'frustratingly':
        src_features, tgt_features, src_feature_names = transform_frustratingly(
            src_features, 
            tgt_features,
            src_feature_names
        )

    # convert target values to multiclass
    src_targets = to_multiclass(src_targets)
    tgt_targets = to_multiclass(tgt_targets)

    # permute
    idxs = np.random.permutation(len(src_features))
    src_features = src_features[idxs]
    src_targets = src_targets[idxs]
    idxs = np.random.permutation(len(tgt_features))
    tgt_features = tgt_features[idxs]
    tgt_targets = tgt_targets[idxs]

    # break into training and validation sets
    src_train_idx = int(train_split * len(src_features))
    src_x_train = src_features[:src_train_idx]
    src_y_train = src_targets[:src_train_idx]
    src_x_val = src_features[src_train_idx:]
    src_y_val = src_targets[src_train_idx:]

    tgt_train_idx = int(train_split * len(tgt_features))
    tgt_x_train = tgt_features[:tgt_train_idx]
    tgt_y_train = tgt_targets[:tgt_train_idx]
    tgt_x_val = tgt_features[tgt_train_idx:]
    tgt_y_val = tgt_targets[tgt_train_idx:]

    # if a maximum number of training samples is given, subselect that here
    if max_src_train_samples is not None:
        src_x_train = src_x_train[:max_src_train_samples]
        src_y_train = src_y_train[:max_src_train_samples]
    if max_tgt_train_samples is not None:
        tgt_x_train = tgt_x_train[:max_tgt_train_samples]
        tgt_y_train = tgt_y_train[:max_tgt_train_samples]

    # two options:
    # 1. combined normalization
    # normalize features based on the combined source and target features
    # the reason for this is that if you normalize the source and target 
    # separately, different unnormalized values might correspond to the 
    # same normalized value, and this seems undesireable
    # 2. individual normalization
    # normalize source and target separately
    # I think this is likely better just from looking at some simple cases
    src_x_train, tgt_x_train, src_x_val, tgt_x_val, stats = normalize(
        src_x_train,
        tgt_x_train,
        src_x_val,
        tgt_x_val,
        mode='combined'
    )

    return dict(
        src_x_train=src_x_train,
        src_y_train=src_y_train,
        src_x_val=src_x_val,
        src_y_val=src_y_val,
        tgt_x_train=tgt_x_train,
        tgt_y_train=tgt_y_train,
        tgt_x_val=tgt_x_val,
        tgt_y_val=tgt_y_val,
        stats=stats,
        feature_names=src_feature_names,
        target_names=src_target_names
    )

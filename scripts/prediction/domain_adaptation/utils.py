
from collections import defaultdict
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics
import sys

def maybe_mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)

def compute_n_batches(n_samples, batch_size):
    n_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches

def compute_batch_idxs(start, batch_size, size, fill='random'):
    if start >= size:
        return list(np.random.randint(low=0, high=size, size=batch_size))
    
    end = start + batch_size

    if end <= size:
        return list(range(start, end))

    else:
        base_idxs = list(range(start, size))
        if fill == 'none':
            return base_idxs
        elif fill == 'random':
            remainder = end - size
            idxs = list(np.random.randint(low=0, high=size, size=remainder))
            return base_idxs + idxs
        else:
            raise ValueError('invalid fill: {}'.format(fill))

def listdict2dictlist(lst):
    dictlist = defaultdict(list)
    for d in lst:
        for k,v in d.items():
            dictlist[k].append(v)
    return dictlist

def process_stats(stats, metakeys=[], score_key='tgt_loss'):
    # actual stats stored one level down
    res = stats['stats']

    # result dictionaries
    train = defaultdict(list)
    val = defaultdict(list)

    # aggregate
    for epoch in res.keys():

        dictlist = listdict2dictlist(res[epoch]['train'])
        for (k,v) in dictlist.items():
            train[k].append(np.mean(v))

        dictlist = listdict2dictlist(res[epoch]['val'])
        for (k,v) in dictlist.items():
            val[k].append(np.mean(v))
        
    score = np.max(val[score_key])
    ret = dict(train=train, val=val, score=score)
    for key in metakeys:
        ret[key] = stats[key]
    return ret


def classification_score(probs, y):
    # the reason to take the argmax here is that for the ngsim data 
    # there are no continous data, but for simulated data the values might 
    # be continous and if that's the case then it will throw an error
    # this is also why we pass the positive class probability 
    for v in y:
        if v[0] != 0 and v[0] != 1:
            y = np.argmax(y, axis=-1)
            probs = probs[:,1]
            break
    
    prc_auc = sklearn.metrics.average_precision_score(y, probs)
    roc_auc = sklearn.metrics.roc_auc_score(y, probs)
    brier = np.mean((probs - y) ** 2)
    return prc_auc, roc_auc, brier

def evaluate(model, dataset):
    src_info = model.evaluate(dataset.xs, dataset.ys, np.zeros(len(dataset.xs)), mode='src')
    tgt_info = model.evaluate(dataset.xt, dataset.yt, np.ones(len(dataset.xt)), mode='tgt')
    return dict(src_info=src_info, tgt_info=tgt_info)

def report(train_info, val_info, extra_val_info=None):
    print('\n')
    print('---------------' * 2)
    for key in ['task_loss', 'prc_auc', 'roc_auc', 'brier']:
        print('train src {}: {:.4f}'.format(key, train_info['src_info'][key]))
        print('train tgt {}: {:.4f}'.format(key, train_info['tgt_info'][key]))
        print('val src {}: {:.4f}'.format(key, val_info['src_info'][key]))
        print('val tgt {}: {:.4f}'.format(key, val_info['tgt_info'][key]))
        if extra_val_info is not None:
            print('extra val src {}: {:.4f}'.format(key, extra_val_info['src_info'][key]))
            print('extra val tgt {}: {:.4f}'.format(key, extra_val_info['tgt_info'][key]))
        print('****************')
    print('----------------' * 2)

def to_multiclass(y):
    ret = np.zeros((len(y), 2))
    ret[:,0] = 1-y
    ret[:,1] = y
    return ret

def load_features_targets_weights(f, target_idx, debug_size=None):
    n_samples = len(f['risk/features'])
    debug_size = n_samples if debug_size is None else debug_size

    features = f['risk/features'][:debug_size]
    targets = f['risk/targets'][:debug_size,:,target_idx]

    if 'risk/weights' in f.keys():
        weights = f['risk/weights'][:debug_size].flatten()
    else:
        weights = np.ones(n_samples)

    return features, targets, weights

def load_single_dataset(
        filepath,
        debug_size=None,
        target_idx=2,
        timestep=-1,
        start_y_timestep=101,
        end_y_timestep=None,
        remove_early_collision_idx=0):

    file = h5py.File(filepath, 'r')
    feature_names = file['risk'].attrs['feature_names']
    target_names = file['risk'].attrs['target_names'][target_idx]
    x, y, w = load_features_targets_weights(file, target_idx, debug_size)

    # convert target values from timeseries to single value
    ## optionally remove instances where there's an early collision
    if remove_early_collision_idx > 0:
        valid_idxs = np.where(y[:,:remove_early_collision_idx].sum(1) == 0)[0]
        x = x[valid_idxs]
        y = y[valid_idxs]
        w = w[valid_idxs]
    ## sum across timesteps because each timestep is already the probability
    ## of a collision occuring at that timestep, where the probability at 
    ## each timestep is mutually exclusive
    end_y_timestep = y.shape[1] if end_y_timestep is None else end_y_timestep
    y = y[:,start_y_timestep:end_y_timestep].sum(1)

    # select the timestep of the features (a single one for now)
    if len(x.shape) > 2:
        x = x[:,timestep]

    # convert y to array format
    y = to_multiclass(y)

    return dict(
        x=x, 
        y=y, 
        w=w, 
        feature_names=feature_names, 
        target_names=target_names
    )

def align_features_targets(src, tgt):
    # target values should just match
    assert src['target_names'] == tgt['target_names']

    # subselect src features to only those features also in the target set
    # need to do this in a manner such that the features align
    # accomplish this by iterating the target feature names
    # finding the src feature name that matches the current target
    # and add its index if it exists
    keep_idxs = []
    for tgt_name in tgt['feature_names']:
        for i, src_name in enumerate(src['feature_names']):
            if src_name == tgt_name:
                keep_idxs.append(i)
                break
    # subselect src features
    assert all(src['feature_names'][keep_idxs] == tgt['feature_names'])
    src['feature_names'] = src['feature_names'][keep_idxs]
    src['x'] = src['x'][...,keep_idxs]

    return src, tgt

def find_n_pos_idx(x, n):
    count, i = 0, 0
    for i, v in enumerate(x):
        if v > 0:
            count += 1
        if count > n:
            break
    return i

def train_val_test_split(d, train_split, max_train_pos=None):
    n_samples = len(d['x'])
    tr_idx = int(train_split * n_samples)
    
    x_tr = d['x'][:tr_idx]
    y_tr = d['y'][:tr_idx]
    w_tr = d['w'][:tr_idx]
    if max_train_pos is not None:
        idx = find_n_pos_idx(y_tr[:,1], max_train_pos)
        x_tr = x_tr[:idx]
        y_tr = y_tr[:idx]
        w_tr = w_tr[:idx]

    val_split = (1 - train_split) / 2.
    val_idx = tr_idx + int(val_split * n_samples)
    x_val = d['x'][tr_idx:val_idx]
    y_val = d['y'][tr_idx:val_idx]
    w_val = d['w'][tr_idx:val_idx]

    x_te = d['x'][val_idx:]
    y_te = d['y'][val_idx:]
    w_te = d['w'][val_idx:]

    d.update(dict(
        x_train=x_tr,
        y_train=y_tr,
        w_train=w_tr,
        x_val=x_val,
        y_val=y_val,
        w_val=w_val,
        x_test=x_te,
        y_test=y_te,
        w_test=w_te
        )
    )
    return d

def normalize_composite(src, tgt):
    # count samples for weighting the respective means
    n_src = len(src['x_train'])
    n_tgt = len(tgt['x_train'])
    n = n_src + n_tgt
    src_ratio = n_src / n
    tgt_ratio = n_tgt / n

    # compute mean by weighting the source and target means by their sizes
    src_mean = np.mean(src['x_train'], axis=0, keepdims=True)
    tgt_mean = np.mean(tgt['x_train'], axis=0, keepdims=True)
    mean = src_ratio * src_mean + tgt_ratio * tgt_mean

    # center the features and compute std
    src['x_train'] -= mean
    tgt['x_train'] -= mean
    src_std = np.std(src['x_train'], axis=0, keepdims=True) 
    tgt_std = np.std(tgt['x_train'], axis=0, keepdims=True)
    std = (src_std * src_ratio + tgt_std * tgt_ratio) + 1e-8

    # finish normalizing train, and normalize val and test
    src['x_train'] /= std
    tgt['x_train'] /= std

    src['x_val'] = (src['x_val'] - mean) / std
    src['x_test'] = (src['x_test'] - mean) / std
    tgt['x_val'] = (tgt['x_val'] - mean) / std
    tgt['x_test'] = (tgt['x_test'] - mean) / std

    src['mean'] = tgt['mean'] = mean
    src['std'] = tgt['std'] = std

    return src, tgt

def normalize_individual(d):
    mean = np.mean(d['x_train'], axis=0, keepdims=True)
    d['x_train'] -= mean
    std = np.std(d['x_train'], axis=0, keepdims=True) + 1e-8
    d['x_train'] /= std

    d['x_val'] = (d['x_val'] - mean) / std
    d['x_test'] = (d['x_test'] - mean) / std
    d['mean'] = mean
    d['std'] = std
    return d

def normalize(src, tgt, mode):
    if mode == 'individual':
        src = normalize_individual(src)
        tgt = normalize_individual(tgt)
    elif mode == 'composite':
        src, tgt = normalize_composite(src, tgt)
    return src, tgt

def load_data(
        src_filepath,
        tgt_filepath,
        debug_size=None,
        target_idx=2,
        timestep=-1,
        start_y_timestep=101,
        end_y_timestep=None,
        remove_early_collision_idx=0,
        src_train_split=.8,
        tgt_train_split=.5,
        n_pos_tgt_train_samples=None,
        normalize_mode='composite'):
    # load in the datasets
    src = load_single_dataset(
        src_filepath,
        debug_size,
        target_idx,
        timestep,
        start_y_timestep,
        end_y_timestep,
        remove_early_collision_idx
    )
    tgt = load_single_dataset(
        tgt_filepath,
        debug_size,
        target_idx,
        timestep,
        start_y_timestep,
        end_y_timestep,
        remove_early_collision_idx
    )

    # align src and tgt target and feature values
    src, tgt = align_features_targets(src, tgt)

    # split each into train, val, test sets
    src = train_val_test_split(src, src_train_split)
    tgt = train_val_test_split(tgt, tgt_train_split, n_pos_tgt_train_samples)

    # normalize the datasets
    src, tgt = normalize(src, tgt, normalize_mode)
    
    return src, tgt

if __name__ == '__main__':
    src_filepath = '../../../data/datasets/nov/subselect_proposal_prediction_data.h5'
    tgt_filepath = '../../../data/datasets/nov/bn_train_data.h5'
    src, tgt = load_data(src_filepath, tgt_filepath)

"""
This scripts selects same from a dataset that are associated with a likelihood
weight that is not 1. This task is performed after dataset collection for now.
"""
import argparse
import h5py
import numpy as np
import os

def copy_attrs(src, dest):
    for k in src.keys():
        try:
            v = src[k]
        except Exception as e:
            print('exception occurred during transfer of key: {}; ignoring'.format(k))
        else:
            if k.startswith('utf8_'):
                k = k.replace('utf8_', '')
                v = ''.join(chr(i) for i in v)
            elif isinstance(v, np.generic):
                v = np.asscalar(v)
            dest[k] = v

    return dest

def select_vehicle_samples(
        input_filepath,
        output_filepath,
        start_veh_idx=60,
        end_veh_idx=70,
        batch_size=1000):
    infile = h5py.File(input_filepath, 'r')
    outfile = h5py.File(output_filepath, 'w')

    # find proposal indices
    weights = infile['risk/weights']
    bidxs = [0] + list(np.copy(infile['risk/batch_idxs'].value[:-1]))
    keep_idxs = []
    for bidx in bidxs:
        keep_idxs += list(range(bidx + start_veh_idx, bidx + end_veh_idx))

    keep_idxs = np.array(keep_idxs)
    nsamples = len(keep_idxs)
    _, timesteps, feature_dim = infile['risk/features'].shape
    _, target_timesteps, target_dim = infile['risk/targets'].shape

    outfile.create_dataset("risk/features", (nsamples, timesteps, feature_dim))
    outfile.create_dataset("risk/targets", (nsamples, target_timesteps, target_dim))

    nbatches = int(nsamples / float(batch_size))
    if nsamples % batch_size != 0:
        nbatches += 1
    for bidx in range(nbatches):
        print('batch: {} / {}'.format(bidx, nbatches))
        s = bidx * batch_size
        e = s + batch_size
        idxs = keep_idxs[s:e]
        outfile['risk/features'][s:e,:,:] = infile['risk/features'][idxs,:,:]
        outfile['risk/targets'][s:e,:] = infile['risk/targets'][idxs,:]
        
    outfile['risk/weights'] = weights.value[keep_idxs]

    # metadata
    outfile['risk/seeds'] = infile['risk/seeds'].value
    outfile['risk/batch_idxs'] = np.arange(len(keep_idxs)).reshape(-1, 1)
    copy_attrs(infile['risk'].attrs, outfile['risk'].attrs)

    infile.close()
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_filepath', default='', type=str,
                            help="filepath to original dataset")
    parser.add_argument('--subselect_filepath', 
                            default='', type=str,
                            help="filepath to output dataset with relevant vehicles")
    args = parser.parse_args()

    select_vehicle_samples(
        args.dataset_filepath, 
        args.subselect_filepath
    )
    

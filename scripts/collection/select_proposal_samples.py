"""
This scripts selects same from a dataset that are associated with a likelihood
weight that is not 1. This task is performed after dataset collection for now.
"""
import h5py
import numpy as np

def select_proposal_samples(input_filepath, output_filepath):
    infile = h5py.File(input_filepath, 'r')
    outfile = h5py.File(output_filepath, 'w')

    # find proposal indices
    weights = infile['risk/weights']
    prop_idxs = np.where(weights[:,0] != 1.)[0]
    
    # select the targets, features, weights from the proposal network
    outfile['risk/features'] = infile['risk/features'].value[prop_idxs]
    outfile['risk/targets'] = infile['risk/targets'].value[prop_idxs]
    outfile['risk/weights'] = weights.value[prop_idxs]

    # metadata
    outfile['risk/seeds'] = infile['risk/seeds'].value
    outfile['risk/batch_idxs'] = np.arange(len(prop_idxs)).reshape(-1, 1)
    outfile['risk'].attrs['feature_names'] = infile['risk'].attrs['feature_names']

    infile.close()
    outfile.close()

if __name__ == '__main__':
    input_filepath = '../../data/datasets/risk_1_lane_prop_5_sec.h5'
    output_filepath = '../../data/datasets/proposal_risk_1_lane_5_sec.h5'
    select_proposal_samples(input_filepath, output_filepath)
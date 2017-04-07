
import numpy as np
import h5py

def subselect_dataset(input_filepath, output_filepath, min_num_samples=500000,
        max_num_samples=1000000, batch_size=10000):
    infile = h5py.File(input_filepath, 'r')
    outfile = h5py.File(output_filepath, 'w')

    num_samples, timesteps, feature_dim = infile['risk/features'].shape
    _, target_dim = infile['risk/targets'].shape

    outfile.create_dataset("risk/features", 
        (max_num_samples, feature_dim), chunks=(100, feature_dim))
    outfile.create_dataset("risk/targets", 
        (max_num_samples, target_dim), chunks=(100, target_dim))

    outfile['risk/features'][:min_num_samples] = infile['risk/features'][:min_num_samples, 0]
    outfile['risk/targets'][:min_num_samples] = infile['risk/targets'][:min_num_samples]

    num_batches = int(num_samples / batch_size)
    if num_batches % batch_size != 0:
        num_batches += 1

    samples_count = min_num_samples
    for bidx in range(num_batches):
        if bidx % 100 == 0:
            print('samples: {} / {}'.format(samples_count, max_num_samples))
        s = bidx * batch_size
        e = s + batch_size

        batch_targets = infile['risk/targets'][s:e]
        batch_features = infile['risk/features'][s:e]

        idxs = np.where(np.sum(batch_targets, axis=1) > 0)[0]
        
        count_after = len(idxs) + samples_count
        if count_after > max_num_samples:
            idxs = idxs[:(max_num_samples - samples_count)]

        valid_count = len(idxs)

        s = samples_count
        e = s + valid_count

        outfile['risk/targets'][s:e] = batch_targets[idxs]
        outfile['risk/features'][s:e] = batch_features[idxs,0,:]

        samples_count += valid_count
        if samples_count > max_num_samples:
            break

    if samples_count < max_num_samples:
        outfile['risk/features'].resize((samples_count, feature_dim))
        outfile['risk/targets'].resize((samples_count, target_dim))

    infile.close()
    outfile.close()

if __name__ == '__main__':
    input_filepath = '../../data/datasets/april/risk_.1_sec_large.h5'
    output_filepath = '../../data/datasets/april/reduced_risk_.1_sec_large.h5'
    subselect_dataset(input_filepath, output_filepath)
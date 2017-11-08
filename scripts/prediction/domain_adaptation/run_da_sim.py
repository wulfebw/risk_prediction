
import argparse
from collections import defaultdict
import numpy as np
import os
import tensorflow as tf

from dann import DANN
from domain_adaptation_dataset import DomainAdaptationDataset
import utils
import visualization_utils

def build_datasets(src, tgt, batch_size):
    datasets = []
    for split in ['train', 'val', 'test']:
        datasets.append(
            DomainAdaptationDataset(
                src['x_{}'.format(split)],
                src['y_{}'.format(split)],
                src['w_{}'.format(split)],
                tgt['x_{}'.format(split)],
                tgt['y_{}'.format(split)],
                tgt['w_{}'.format(split)],
                batch_size=batch_size
            )
        )
    
    return datasets[0], datasets[1], datasets[2]

def run_training(
        dataset, 
        val_dataset,
        network_size,
        dropout_keep_prob,
        learning_rate,
        lambda_final,
        n_updates=10000,
        batch_size=100):

    # unpack shapes
    n_src_samples, input_dim = dataset.xs.shape
    n_tgt_samples, _ = dataset.xt.shape

    # decide the number of epochs so as to achieve a specified number of updates
    n_samples = max(n_src_samples, n_tgt_samples)
    updates_per_epoch = (n_samples / batch_size)
    n_epochs = int(n_updates // updates_per_epoch)

    # tf reset
    tf.reset_default_graph()
    with tf.Session() as sess:

        # build the model and initialize
        model = DANN(
            input_dim=input_dim, 
            output_dim=2,
            lambda_final=lambda_final,
            lambda_steps=n_updates / 4,
            dropout_keep_prob=dropout_keep_prob,
            learning_rate=learning_rate,
            encoder_hidden_layer_dims=network_size,
            classifier_hidden_layer_dims=()
        )

        # tf initialization
        sess.run(tf.global_variables_initializer())

        # train the model
        val_every = 1 # max(1, int(n_epochs / 10))
        stats = model.train(
            dataset, 
            val_dataset=val_dataset, 
            val_every=val_every, 
            n_epochs=n_epochs
        )

    return stats

def hyperparam_search(
        src, 
        tgt, 
        mode,
        network_sizes,
        dropout_keep_probs,
        learning_rates,
        n_itr,
        stats_filepath_template,
        batch_size=100):
    
    # set values conditional on mode
    if mode == 'with_adapt':
        lambda_final = 0.5
    elif mode == 'without_adapt':
        lambda_final = 0.
    elif mode == 'target_only':
        src = tgt
        lambda_final = 0.

    # build datasets
    dataset, val_dataset, test_dataset = build_datasets(src, tgt, batch_size)

    # track stats
    for itr in range(n_itr):
        stats = dict()
        stats['itr'] = itr
        stats['network_size'] = np.random.choice(network_sizes)
        stats['dropout_keep_prob'] = np.random.choice(dropout_keep_probs)
        stats['learning_rate'] = np.random.choice(learning_rates)

        stats['stats'] = run_training(
            dataset, 
            val_dataset, 
            stats['network_size'],
            stats['dropout_keep_prob'],
            stats['learning_rate'],
            lambda_final,
            batch_size=batch_size
        )

        stats = utils.process_stats(stats)
        stats_filepath = stats_filepath_template.format(stats['score'], itr)
        np.save(stats_filepath, stats)

def main(
        mode='with_adapt',
        source_filepath='../../../data/datasets/nov/subselect_proposal_prediction_data.h5',
        target_filepath='../../../data/datasets/nov/bn_train_data.h5',
        results_dir='../../../data/datasets/nov/hyperparam_search',
        visualize=False,
        vis_dir='../../../data/visualizations/domain_adaptation',
        batch_size=1000,
        debug_size=1000000):
    
    utils.maybe_mkdir(results_dir)

    src, tgt = utils.load_data(
        source_filepath, 
        target_filepath, 
        debug_size=debug_size
    )
    hyperparam_search(
        src, 
        tgt, 
        mode, 
        network_sizes=[
            (512, 512, 256, 256, 128, 64),
            (512, 256, 128, 64),
            (128, 64)
        ],
        dropout_keep_probs=[.5, .75, 1.],
        learning_rates=[1e-4, 2e-4, 5e-4, 8e-4],
        n_itr=20,
        stats_filepath_template=os.path.join(
            results_dir, '{:.4f}_itr_{}_' + '{}.npy'.format(mode))
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mode parser')
    parser.add_argument('--mode', type=str, default='with_adapt')
    args = parser.parse_args()
    stats = main(mode=args.mode)

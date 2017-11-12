
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
        encoder_size,
        classifier_size,
        dropout_keep_prob,
        learning_rate,
        lambda_final,
        src_only_adversarial,
        da_mode,
        n_epochs=100,
        val_every=1,
        batch_size=100):

    # unpack shapes
    n_src_samples, input_dim = dataset.xs.shape
    n_tgt_samples, _ = dataset.xt.shape

    # decide number of lambda steps based on updates per epoch
    n_samples = max(n_src_samples, n_tgt_samples)
    updates_per_epoch = (n_samples / batch_size)
    lambda_steps = updates_per_epoch * n_epochs / 5

    # tf reset
    tf.reset_default_graph()
    with tf.Session() as sess:

        # build the model and initialize
        model = DANN(
            input_dim=input_dim, 
            output_dim=2,
            lambda_final=lambda_final,
            lambda_steps=lambda_steps,
            dropout_keep_prob=dropout_keep_prob,
            learning_rate=learning_rate,
            encoder_hidden_layer_dims=encoder_size,
            classifier_hidden_layer_dims=classifier_size,
            src_only_adversarial=src_only_adversarial,
            shared_classifier=True,
            da_mode=da_mode
        )

        # tf initialization
        sess.run(tf.global_variables_initializer())

        # train the model
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
        encoder_sizes,
        classifier_sizes,
        dropout_keep_probs,
        learning_rates,
        n_itr,
        stats_filepath_template,
        src_only_adversarials=[False],
        batch_size=100,
        n_epochs=100):
    
    # set values conditional on mode
    if mode == 'with_adapt':
        lambda_final = 0.1
        da_mode = 'unsupervised'
    elif mode == 'with_sup_adapt':
        lambda_final = 0.1
        da_mode = 'supervised'
    elif mode == 'with_uni_adapt':
        lambda_final = 0.1
        da_mode = 'unsupervised'
        src_only_adversarials = [True]
    elif mode == 'without_adapt':
        lambda_final = 0.
        da_mode = 'unsupervised'
    elif mode == 'target_only':
        src = tgt
        lambda_final = 0.
        da_mode = 'unsupervised'

    # build datasets
    dataset, val_dataset, test_dataset = build_datasets(src, tgt, batch_size)

    # track stats
    for itr in range(n_itr):
        stats = dict()
        stats['itr'] = itr
        stats['encoder_size'] = np.random.choice(encoder_sizes)
        stats['classifier_size'] = np.random.choice(classifier_sizes)
        stats['dropout_keep_prob'] = np.random.choice(dropout_keep_probs)
        stats['learning_rate'] = np.random.choice(learning_rates)
        stats['src_only_adversarial'] = np.random.choice(src_only_adversarials)

        stats['stats'] = run_training(
            dataset, 
            val_dataset, 
            stats['encoder_size'],
            stats['classifier_size'],
            stats['dropout_keep_prob'],
            stats['learning_rate'],
            lambda_final,
            stats['src_only_adversarial'],
            da_mode,
            batch_size=batch_size,
            n_epochs=n_epochs
        )

        stats = utils.process_stats(stats, score_key='pos_ce', agg_fn=np.min, metakeys=[
            'encoder_size', 
            'classifier_size',
            'dropout_keep_prob', 
            'learning_rate',
            'src_only_adversarial'
        ])

        stats_filepath = stats_filepath_template.format(stats['score'], itr)
        np.save(stats_filepath, stats)
        print(
            '\nencoder size: {}\nclassifier size: {}\ndropout: {:.5f}\nlr: {:.5f}\nsrc_only_adv: {}\nscore: {:.5f}'.format(
                stats['encoder_size'], 
                stats['classifier_size'], 
                stats['dropout_keep_prob'], 
                stats['learning_rate'], 
                stats['src_only_adversarial'],
                stats['score']
            )
        )

def main(
        mode='with_adapt',
        source_filepath='../../../data/datasets/nov/subselect_proposal_prediction_data.h5',
        target_filepath='../../../data/datasets/nov/bn_train_data.h5',
        results_dir='../../../data/datasets/nov/hyperparam_search',
        target_idx=2,
        batch_size=500,
        debug_size=100000,
        n_pos_tgt_train_samples=[1, 10, 25, 50, 75, 100],
        n_tgt_train_samples=None,
        n_epochs=[20, 22, 25, 30, 35, 40]):
    
    utils.maybe_mkdir(results_dir)
    if n_pos_tgt_train_samples is not None:
        n_itr = len(n_pos_tgt_train_samples)
    elif n_tgt_train_samples is not None:
        n_itr = len(n_tgt_train_samples)

    for i in range(n_itr):

        if n_pos_tgt_train_samples is not None:
            src, tgt = utils.load_data(
                source_filepath, 
                target_filepath, 
                debug_size=debug_size,
                remove_early_collision_idx=5,
                n_pos_tgt_train_samples=n_pos_tgt_train_samples[i],
                src_train_split=.9,
                tgt_train_split=2./3,
                target_idx=target_idx
            )
            n_samples = n_pos_tgt_train_samples[i]
        elif n_tgt_train_samples is not None:
            src, tgt = utils.load_data(
                source_filepath, 
                target_filepath, 
                debug_size=debug_size,
                remove_early_collision_idx=5,
                n_tgt_train_samples=n_tgt_train_samples[i],
                src_train_split=.9,
                tgt_train_split=2/3.,
                target_idx=target_idx
            )
            n_samples = n_tgt_train_samples[i]

        template = os.path.join(
                results_dir,
                '{}_'.format(n_samples) + '{:.4f}_itr_{}_' + '{}.npy'.format(mode))
        hyperparam_search(
            src, 
            tgt, 
            mode,
            encoder_sizes=[
                (512, 256, 128, 64),
                (256, 128, 64),
                (128, 64)
            ],
            classifier_sizes=[
                (),
                (64,),
                (64,64)
            ],
            dropout_keep_probs=np.linspace(.5,1,200),
            learning_rates=np.linspace(1e-4,1e-3,200),
            n_itr=30,
            stats_filepath_template=template,
            n_epochs=n_epochs[i]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mode parser')
    parser.add_argument('--mode', type=str, default='with_adapt')
    parser.add_argument('--src_filepath', type=str, 
        default='../../../data/datasets/nov/sim2real.h5')
    parser.add_argument('--tgt_filepath', type=str, 
        default='../../../data/datasets/nov/ngsim_20_sec_1_feature_timesteps_traj_1_3_2.h5')
    parser.add_argument('--target_idx', type=int, default=4)
    args = parser.parse_args()
    stats = main(
        mode=args.mode, 
        source_filepath=args.src_filepath,
        target_filepath=args.tgt_filepath,
        target_idx=args.target_idx)

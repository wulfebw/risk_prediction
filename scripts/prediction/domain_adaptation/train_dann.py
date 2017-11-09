

import numpy as np
import os
import tensorflow as tf

from dann import DANN
from domain_adaptation_dataset import DomainAdaptationDataset
import utils
import visualization_utils

source_filepath = '../../../data/datasets/nov/bn_train_data.h5'
target_filepath = '../../../data/datasets/nov/bn_train_data.h5'
vis_dir = '/Users/wulfebw/Desktop/tmp/'
debug_size = 100000
batch_size = 500
n_pos_tgt_train = 5

src, tgt = utils.load_data(
    source_filepath, 
    target_filepath, 
    debug_size=debug_size,
    remove_early_collision_idx=0,
    src_train_split=.5,
    tgt_train_split=.5,
    n_pos_tgt_train_samples=n_pos_tgt_train,
)

print(src['y_train'].shape)
print(src['y_val'].shape)
print(tgt['y_train'].shape)
print(tgt['y_val'].shape)

datasets = []
for split in ['train', 'val']:
    datasets.append(DomainAdaptationDataset(
        src['x_{}'.format(split)],
        src['y_{}'.format(split)],
        src['w_{}'.format(split)],
        tgt['x_{}'.format(split)],
        tgt['y_{}'.format(split)],
        tgt['w_{}'.format(split)],
        batch_size=batch_size
    ))
dataset, val_dataset = datasets

tf.reset_default_graph()
with tf.Session() as sess:
    model = DANN(
        input_dim=src['x_train'].shape[-1], 
        output_dim=2,
        lambda_final=.5,
        lambda_steps=200,
        dropout_keep_prob=1.,
        learning_rate=1e-3,
        encoder_hidden_layer_dims=(64,),
        classifier_hidden_layer_dims=(),
        src_only_adversarial=False,
        shared_classifier=True
    )
    sess.run(tf.global_variables_initializer())
    model.train(
        dataset, 
        val_dataset=val_dataset, 
        val_every=2, 
        n_epochs=20, 
        verbose=True
    )
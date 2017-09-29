
import numpy as np
import os
import tensorflow as tf

import utils
import rnn

def main():
    dataset_filepath = '../../data/datasets/ngsim_feature_trajectories.h5'
    binedges = [10,15,25,50]
    max_len = 100
    data = utils.load_ngsim_trajectory_data(
        dataset_filepath,
        binedges=binedges,
        max_len=max_len,
        max_samples=123,
        train_ratio=.9
    )
    print(len(data['train_x']))

    exp_dir = '../../data/experiments/imputation'
    utils.maybe_mkdir(exp_dir)

    model = rnn.RNN(
        name='supervised_imputation',
        input_dim=data['train_x'].shape[2],
        hidden_dim=128,
        max_len=max_len,
        output_dim=len(binedges),
        batch_size=50,
        learning_rate=.0005,
        dropout_keep_prob=.75
    )
    writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train'))
    val_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'val'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(
            data, 
            n_epochs=100000,
            writer=writer,
            val_writer=val_writer
        )

if __name__ == '__main__':
    main()


import collections
import numpy as np
import os
import tensorflow as tf

from dann import DANN
import utils
import visualization_utils

def run_training(
        dataset, 
        val_dataset,
        n_updates=5000,
        lambda_final=1.,
        lambda_steps=200,
        batch_size=100):

    # unpack shapes
    n_src_samples, input_dim = dataset.xs.shape
    n_tgt_samples, _ = dataset.xt.shape

    # decide the number of epochs so as to achieve a specified number of updates
    n_samples = max(n_src_samples, n_tgt_samples)
    updates_per_epoch = (n_samples / batch_size) * 2
    n_epochs = int(n_updates // updates_per_epoch)

    # tf setup
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # build the model and initialize
    model = DANN(
        input_dim=input_dim, 
        output_dim=2,  
        lambda_final=lambda_final,
        lambda_steps=lambda_steps,
        dropout_keep_prob=.6,
        encoder_hidden_layer_dims=(256,256,256,256)
    )

    sess.run(tf.global_variables_initializer())

    # train the model
    model.train(dataset, n_epochs=n_epochs)

    # evaluate the model
    train_info = utils.evaluate(model, dataset)
    val_info = utils.evaluate(model, val_dataset)

    # report
    utils.report(train_info, val_info)

    return dict(train_info=train_info, val_info=val_info)

def main(
        visualize=True,
        batch_size=100,
        vis_dir='../../../data/visualizations/domain_adaptation',
        output_filepath_template='../../../data/datasets/da_results_*_{}.npy',
        source_filepath='../../../data/datasets/oct/bn_train_data.h5',
        target_filepath='../../../data/datasets/ngsim_5_sec_40_feature_timesteps.h5',
        n_tgt_train_samples = [500, 1000, 5000, 10000, 20000, None],
        n_src_train_samples = [None, None, None, None, None, None],
        mode='with_adapt',
        seeds=[1,2,3,4,5,6]):
    
    # set output filepath template based on mode
    output_filepath_template = output_filepath_template.replace('*', mode)

    # modes
    if mode == 'with_adapt':
        lambda_final = 1.
    elif mode == 'without_adapt':
        lambda_final = 0.
    elif mode == 'target_only':
        n_tgt_train_samples = [int(v * .5) for v in n_tgt_train_samples if v != None]
        n_tgt_train_samples.append(None)
        n_src_train_samples = n_tgt_train_samples
        source_filepath = target_filepath
        lambda_final = 0
    else:
        raise(ValueError('invalid mode: {}'.format(mode)))

    # debug
    # source_filepath = '../../../data/datasets/debug_source.h5'
    # target_filepath = '../../../data/datasets/debug_target.h5'
    
    n_sample_sizes = len(n_tgt_train_samples)

    infos = dict()
    for i in range(n_sample_sizes):

        # set the seed
        np.random.seed(seeds[i])

        # load the data for this size
        data = utils.load_data(
            source_filepath, 
            target_filepath,
            max_tgt_train_samples=n_tgt_train_samples[i],
            max_src_train_samples=n_src_train_samples[i],
            timestep=-1
        )
        
        if visualize:
            utils.maybe_mkdir(vis_dir)
            visualization_utils.visualize(data, vis_dir)

        # build datasets
        dataset, val_dataset = utils.build_datasets(data, batch_size)

        # update n_tgt_samples in case fewer than requested were loaded
        n_tgt_train_samples[i] = len(dataset.xt)

        # report training size
        print('training with {} target samples'.format(n_tgt_train_samples[i]))

        # train
        cur_size = n_tgt_train_samples[i]
        if mode == 'target_only' and i != n_sample_sizes - 1:
            cur_size *= 2
        infos[cur_size] = run_training(
            dataset, 
            val_dataset, 
            batch_size=batch_size,
            lambda_final=lambda_final
        )
        np.save(output_filepath_template.format(n_tgt_train_samples[i]), infos)

if __name__ == '__main__':
    main()

import collections
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(os.path.abspath(path))
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

from prediction import model
from test_config import TestConfig

def generate_data(sigma, n_mc, n_samples, n_timesteps, input_dim):
    x = np.random.randn(n_samples * n_timesteps * input_dim).reshape(
        n_samples, n_timesteps, input_dim)
    y = np.mean(x, axis=(1, 2)).reshape(-1, 1)
    y = np.tile(y, (1, n_mc))
    y += np.random.randn(np.prod(y.shape)).reshape(y.shape) * sigma
    y[y>0] = 1
    y[y<=0] = 0
    y = y.mean(axis=1, keepdims=True)
    return x, y


def test_prediction_across_target_variance(
        val_x, 
        val_y,
        sigma=0,
        n_mc=1,
        hidden_layer_sizes=[32,32], 
        n_epochs=50,
        n_samples=100,
        input_dim=1,
        n_timesteps=2
    ):
    config = TestConfig()
    config.hidden_layer_sizes = hidden_layer_sizes
    config.value_dim = 1
    config.learning_rate = 1e-3
    config.n_epochs = n_epochs
    
    # simple dataset
    x, y = generate_data(sigma, n_mc, n_samples, n_timesteps, input_dim)
    val_losses = []
    with tf.Session() as session:
        predictor = model.LSTMPredictor((input_dim,), config)
        target_placeholder = tf.placeholder(tf.float32, [None, 1], 'target')
        loss = tf.reduce_sum((predictor.vf[-1] - target_placeholder) ** 2)
        opt = tf.train.AdamOptimizer(config.learning_rate)
        train_op = opt.minimize(loss)
        session.run(tf.global_variables_initializer())

        def run_sample(p, x, y, state_in, train=True):
            feed_dict = {
                p.x: x,
                target_placeholder: y,
                p.dropout_keep_prob_ph: 1.,
                p.state_in[0]: state_in[0],
                p.state_in[1]: state_in[1],
            }
            outputs_list = [loss]
            if train:
                outputs_list += [train_op]
            else:
                outputs_list += [p.vf[-1]]
            fetched = session.run(outputs_list, feed_dict=feed_dict)
            
            if train:
                val_loss, _ = fetched
                return val_loss
            else:
                val_loss, val_vf = fetched
                return val_loss, val_vf

        n_val = len(val_x)
        for epoch in range(n_epochs):

            # train
            train_loss_mean = 0
            for sidx in range(n_samples):
                train_loss = run_sample(
                    predictor,
                    x[sidx,:,:], 
                    y[sidx].reshape(1,-1),
                    predictor.state_init
                )
                train_loss_mean += train_loss / n_samples
            
            # val
            val_loss_mean = 0
            for sidx in range(n_val):
                val_loss, val_vf = run_sample(
                        predictor,
                        val_x[sidx,:,:], 
                        val_y[sidx].reshape(1,-1),
                        predictor.state_init,
                        train=False
                    )
                val_loss_mean += val_loss / n_val
                # print('x: {}\ny: {}\ny_pred: {}'.format(
                #     x[sidx,:,:], y[sidx].reshape(1,-1), val_vf))
                # input()
            
            # report, track
            val_losses.append(val_loss_mean)
            print('epoch: {} / {}\ttrain loss: {}\tval loss: {}'.format(
                    epoch, n_epochs, train_loss_mean, val_loss_mean))

    return val_losses   

def plot_losses(filepath, df):

    for sigma in df.sigmas.unique():
        for n_mc in df.n_mc.unique():
            losses = df[(df.n_mc == n_mc) & (df.sigmas == sigma)]['loss']
            print(losses)
            losses = losses.values[0]
            print(losses)
            plt.plot(range(len(losses)), losses, label='n_mc = {}'.format(n_mc))
        plt.xlabel('epochs')
        plt.ylabel('validation loss')
        plt.legend()
        plt.title('Validation Across MC runs for sigma = {}'.format(sigma))
        plt.savefig('../data/sigma_{}.png'.format(sigma))
        plt.clf()

def main():
    tf.set_random_seed(1)
    np.random.seed(1)

    n_timesteps = 4
    input_dim = 2
    n_samples_val = 10000
    n_mc_val = 500

    losses = []
    sigmas = [0, 0.001, 0.01, 0.1, 0.2]
    n_mcs = [1, 5, 50]
    data = collections.defaultdict(list)
    for sigma in sigmas:
        val_x, val_y = generate_data(
            sigma, n_mc_val, n_samples_val, n_timesteps, input_dim)
        for n_mc in n_mcs:
            tf.reset_default_graph()
            loss = test_prediction_across_target_variance(
                val_x,
                val_y,
                sigma=sigma,
                n_mc=n_mc,
                n_epochs=50,
                n_samples=20000,
                input_dim=input_dim,
                n_timesteps=n_timesteps
            )
            data['sigmas'].append(sigma)
            data['n_mc'].append(n_mc)
            data['loss'].append(loss)

    df = pd.DataFrame(data=data)
    filepath = '../data/sigmas_n_mc.pkl'
    df.to_pickle(filepath)
    df = pd.read_pickle(filepath)
    plot_losses('../data/sigmas_n_mc.png', df)


if __name__ == '__main__':
    main()
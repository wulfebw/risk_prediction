
import numpy as np
import os
import sys
import tensorflow as tf
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(os.path.abspath(path))
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

from prediction import model
from test_config import TestConfig

class TestAsyncTD(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()

    def test_full_sequence_prediction(self):
        config = TestConfig()
        config.hidden_layer_sizes = [32,32]
        config.value_dim = 1
        config.learning_rate = 1e-3
        
        # simple dataset, learn to output the sum
        n_samples = 100
        n_timesteps = 4
        input_dim = 1
        x = np.random.rand(n_samples, n_timesteps, input_dim)
        y = np.sum(x, axis=(1, 2)).reshape(-1, 1)

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

            n_epochs = 10
            n_train = int(n_samples * .8)
            n_val = n_samples - n_train
            verbose = False
            for epoch in range(n_epochs):

                # train
                train_loss_mean = 0
                for sidx in range(n_train):
                    train_loss = run_sample(
                        predictor,
                        x[sidx,:,:], 
                        y[sidx].reshape(1,-1),
                        predictor.state_init
                    )
                    train_loss_mean += train_loss / n_train
                
                # val
                val_loss_mean = 0
                for sidx in range(n_val):
                    val_loss, val_vf = run_sample(
                            predictor,
                            x[sidx,:,:], 
                            y[sidx].reshape(1,-1),
                            predictor.state_init,
                            train=False
                        )
                    val_loss_mean += val_loss / n_val
                    # print('x: {}\ny: {}\ny_pred: {}'.format(
                    #     x[sidx,:,:], y[sidx].reshape(1,-1), val_vf))
                    # input()
                
                # report
                if verbose:
                    print('epoch: {} / {}\ttrain loss: {}\tval loss: {}'.format(
                            epoch, n_epochs, train_loss_mean, val_loss_mean))

            self.assertTrue(train_loss_mean < 1e-2)
            self.assertTrue(val_loss_mean < 1e-2)

if __name__ == '__main__':
    unittest.main()
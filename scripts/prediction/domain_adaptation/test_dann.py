
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import unittest

from dann import DANN
from domain_adaptation_dataset import DomainAdaptationDataset
import utils

def generate_data(n_samples, input_dim):
    x = np.random.randn(n_samples, input_dim)
    xsum = np.sum(x, axis=-1)
    y = np.zeros((n_samples, 2))
    zeros = np.where(xsum < 0)[0]
    ones = np.where(xsum > 0)[0]
    y[zeros,0] = 1
    y[ones, 1] = 1
    return x, y

class TestDANN(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        np.random.seed(1)
        tf.set_random_seed(1)
        tf.reset_default_graph()

    def test_fit_simple_dataset(self):

        input_dim = 2
        batch_size = 1000

        sess = tf.InteractiveSession()
        model = DANN(
            input_dim=input_dim, 
            output_dim=2,
            lambda_final=.0,
            lambda_steps=100,
            dropout_keep_prob=1.,
            learning_rate=5e-4,
            encoder_hidden_layer_dims=(64,),
            classifier_hidden_layer_dims=(64,),
            src_only_adversarial=False
        )
        sess.run(tf.global_variables_initializer())

        xs_tr, ys_tr = generate_data(n_samples=1000, input_dim=input_dim)
        xs_val, ys_val = generate_data(n_samples=1000, input_dim=input_dim)
        xt_tr, yt_tr = generate_data(n_samples=1000, input_dim=input_dim)
        xt_val, yt_val = generate_data(n_samples=10000, input_dim=input_dim)

        dataset = DomainAdaptationDataset(
            xs_tr, ys_tr, np.ones(ys_tr.shape[0]),
            xt_tr, yt_tr, np.ones(yt_tr.shape[0]),
            batch_size=batch_size
        )
        
        model.train(dataset, n_epochs=100, verbose=False)
        probs = model.predict(xt_val)
        acc = np.mean(np.equal(np.argmax(probs, axis=-1), np.argmax(yt_val, axis=-1)))
        self.assertTrue(acc > .995)

    def test_fit_simple_dataset_where_source_required(self):

        input_dim = 10
        batch_size = 1000
        n_epochs = 20
        verbose = False
        learning_rate = 1e-3

        xs_tr, ys_tr = generate_data(n_samples=10000, input_dim=input_dim)
        xs_val, ys_val = generate_data(n_samples=10000, input_dim=input_dim)
        xt_tr, yt_tr = generate_data(n_samples=100, input_dim=input_dim)
        xt_val, yt_val = generate_data(n_samples=10000, input_dim=input_dim)

        dataset = DomainAdaptationDataset(
            xs_tr, ys_tr, np.ones(ys_tr.shape[0]),
            xt_tr, yt_tr, np.ones(yt_tr.shape[0]),
            batch_size=batch_size
        )
        val_dataset = DomainAdaptationDataset(
            xs_val, ys_val, np.ones(ys_val.shape[0]),
            xt_val, yt_val, np.ones(yt_val.shape[0]),
            batch_size=batch_size
        )

        with tf.Session() as sess:
            model = DANN(
                input_dim=input_dim, 
                output_dim=2,
                lambda_final=.0,
                lambda_steps=100,
                dropout_keep_prob=1.,
                learning_rate=learning_rate,
                encoder_hidden_layer_dims=(64,),
                classifier_hidden_layer_dims=(),
                src_only_adversarial=False,
                shared_classifier=True
            )
            sess.run(tf.global_variables_initializer())

            model.train(
                dataset, 
                val_dataset=val_dataset, 
                val_every=10, 
                n_epochs=n_epochs, 
                verbose=verbose
            )
            probs = model.predict(xt_val)
            shared_acc = np.mean(np.equal(np.argmax(probs, axis=-1), np.argmax(yt_val, axis=-1)))

        tf.reset_default_graph()
        with tf.Session() as sess:
            model = DANN(
                input_dim=input_dim, 
                output_dim=2,
                lambda_final=.0,
                lambda_steps=100,
                dropout_keep_prob=1.,
                learning_rate=learning_rate,
                encoder_hidden_layer_dims=(64,),
                classifier_hidden_layer_dims=(),
                src_only_adversarial=False,
                shared_classifier=False
            )
            sess.run(tf.global_variables_initializer())
            model.train(
                dataset, 
                val_dataset=val_dataset, 
                val_every=10, 
                n_epochs=n_epochs, 
                verbose=verbose
            )
            probs = model.predict(xt_val)
            private_acc = np.mean(np.equal(np.argmax(probs, axis=-1), np.argmax(yt_val, axis=-1)))
        self.assertTrue(shared_acc > private_acc)

    def test_fit_simple_dataset_where_adaptation_required(self):

        input_dim = 10
        batch_size = 1000
        n_epochs = 20
        verbose = False
        learning_rate = 1e-3

        xs_tr, ys_tr = generate_data(n_samples=10000, input_dim=input_dim)
        xs_val, ys_val = generate_data(n_samples=10000, input_dim=input_dim)
        xt_tr, yt_tr = generate_data(n_samples=100, input_dim=input_dim)
        xt_val, yt_val = generate_data(n_samples=10000, input_dim=input_dim)

        dataset = DomainAdaptationDataset(
            xs_tr, ys_tr, np.ones(ys_tr.shape[0]),
            xt_tr, yt_tr, np.ones(yt_tr.shape[0]),
            batch_size=batch_size
        )
        val_dataset = DomainAdaptationDataset(
            xs_val, ys_val, np.ones(ys_val.shape[0]),
            xt_val, yt_val, np.ones(yt_val.shape[0]),
            batch_size=batch_size
        )

        with tf.Session() as sess:
            model = DANN(
                input_dim=input_dim, 
                output_dim=2,
                lambda_final=.0,
                lambda_steps=100,
                dropout_keep_prob=1.,
                learning_rate=learning_rate,
                encoder_hidden_layer_dims=(64,),
                classifier_hidden_layer_dims=(),
                src_only_adversarial=False,
                shared_classifier=True
            )
            sess.run(tf.global_variables_initializer())

            model.train(
                dataset, 
                val_dataset=val_dataset, 
                val_every=10, 
                n_epochs=n_epochs, 
                verbose=verbose
            )
            probs = model.predict(xt_val)
            shared_acc = np.mean(np.equal(np.argmax(probs, axis=-1), np.argmax(yt_val, axis=-1)))

        tf.reset_default_graph()
        with tf.Session() as sess:
            model = DANN(
                input_dim=input_dim, 
                output_dim=2,
                lambda_final=.0,
                lambda_steps=100,
                dropout_keep_prob=1.,
                learning_rate=learning_rate,
                encoder_hidden_layer_dims=(64,),
                classifier_hidden_layer_dims=(),
                src_only_adversarial=False,
                shared_classifier=False
            )
            sess.run(tf.global_variables_initializer())
            model.train(
                dataset, 
                val_dataset=val_dataset, 
                val_every=10, 
                n_epochs=n_epochs, 
                verbose=verbose
            )
            probs = model.predict(xt_val)
            private_acc = np.mean(np.equal(np.argmax(probs, axis=-1), np.argmax(yt_val, axis=-1)))
        self.assertTrue(shared_acc > private_acc)




if __name__ == '__main__':
    unittest.main()

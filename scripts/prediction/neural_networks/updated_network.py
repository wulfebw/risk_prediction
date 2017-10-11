
import collections
import numpy as np
import os
import sys
import tensorflow as tf

import utils

path = os.path.dirname(os.path.realpath(__file__))
path =  os.path.abspath(os.path.join(path,'..','..',))
sys.path.append(path)
import imputation.utils as rnn_utils

class Network(object):

    def __init__(
            self,
            name,
            input_dim,
            output_dim=2,
            dropout_keep_prob=1.,
            learning_rate=0.001,
            grad_clip=1.,
            is_recurrent=False,
            batch_size=None,
            l2_reg=1e-5):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_keep_prob = dropout_keep_prob
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.is_recurrent = is_recurrent
        assert not (is_recurrent and batch_size is None)
        self.batch_size = batch_size
        self.l2_reg = l2_reg

        self.build_model()

    def build_model(self):
        with tf.variable_scope(self.name):
            self.build_placeholders()
            self.build_network()
            self.build_loss()
            self.build_train_op()
            self.build_summaries()

    def build_placeholders(self):
        if self.is_recurrent:
            input_shape = (self.batch_size, self.max_len, self.input_dim)
        else:
            input_shape = (self.batch_size, self.input_dim)

        self.inputs = tf.placeholder(tf.float32, input_shape, 'inputs')
        self.targets = tf.placeholder(tf.float32, (self.batch_size, self.output_dim), 'targets')
        self.dropout_keep_prop_ph = tf.placeholder_with_default(
            self.dropout_keep_prob, (), 'dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def build_loss(self):
        # compute for each timestep
        data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.targets, logits=self.scores))

        # define variables for model and add regularization loss
        self.var_list = tf.trainable_variables()
        reg_loss = self.l2_reg * tf.reduce_mean([tf.nn.l2_loss(v) for v in self.var_list])

        self.loss = data_loss + reg_loss

        # summaries
        tf.summary.scalar('data_loss', data_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('loss', self.loss)

    def build_train_op(self):
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_vars = optimizer.compute_gradients(self.loss, self.var_list)
        clipped_grads_vars = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v) 
            for (g,v) in grads_vars]
        self.train_op = optimizer.apply_gradients(clipped_grads_vars, global_step=self.global_step)

        # summaries
        tf.summary.scalar('grads_global_norm',
            tf.global_norm([g for (g,_) in grads_vars]))
        tf.summary.scalar('clipped_grads_global_norm',
            tf.global_norm([g for (g,_) in clipped_grads_vars]))
        tf.summary.scalar('vars_global_norm', tf.global_norm(self.var_list))
        tf.summary.scalar('learning_rate', self.learning_rate)

    def build_summaries(self):
        self.summary_op = tf.summary.merge_all()

    def train(
            self, 
            data, 
            n_epochs=100, 
            batch_size=100,
            writer=None, 
            val_writer=None,
            stop_early=False):
        sess = tf.get_default_session()

        if self.batch_size is not None:
            batch_size = self.batch_size
        
        n_samples = len(data['x_train'])
        n_batches = utils.compute_n_batches(n_samples, batch_size)
        n_val_samples = len(data['x_val'])
        n_val_batches = utils.compute_n_batches(n_val_samples, batch_size)

        last_val_losses = collections.deque([np.inf] * 2)
        
        for epoch in range(n_epochs):

            # shuffle train set
            idxs = np.random.permutation(len(data['x_train']))
            data['x_train'] = data['x_train'][idxs]
            data['y_train'] = data['y_train'][idxs]

            # train
            total_loss = 0
            for bidx in range(n_batches):
                idxs = utils.compute_batch_idxs(bidx * batch_size, batch_size, n_samples)
                feed_dict = {
                    self.inputs:data['x_train'][idxs],
                    self.targets:data['y_train'][idxs],
                }
                outputs_list = [self.loss, self.summary_op, self.global_step, self.train_op]
                loss, summary, step, _ = sess.run(outputs_list, feed_dict=feed_dict)
                total_loss += loss
                if writer is not None:
                    writer.add_summary(summary, step)
                sys.stdout.write('\repoch: {} / {} batch: {} / {} loss: {}'.format(
                    epoch+1, n_epochs, bidx+1, n_batches, 
                    total_loss / (bidx+1)))
            print('\n')

            # val
            total_loss = 0
            for bidx in range(n_val_batches):
                s = bidx * batch_size
                e = s + batch_size
                idxs = utils.compute_batch_idxs(bidx * batch_size, batch_size, n_val_samples)
                feed_dict = {
                    self.inputs:data['x_val'][idxs],
                    self.targets:data['y_val'][idxs],
                    self.dropout_keep_prop_ph: 1.
                }
                outputs_list = [self.loss, self.summary_op, self.global_step]
                loss, summary, step = sess.run(outputs_list, feed_dict=feed_dict)
                total_loss += loss
                if val_writer is not None:
                    val_writer.add_summary(summary, step)
                sys.stdout.write('\rval epoch: {} / {} batch: {} / {} loss: {}'.format(
                    epoch+1, n_epochs, bidx+1, n_val_batches, 
                    total_loss / (bidx+1)))
            print('\n')
            if stop_early:
                if all(total_loss > v for v in last_val_losses):
                    break
                last_val_losses.popleft()
                last_val_losses.append(total_loss)

    def predict(self, inputs, batch_size=100):
        if self.batch_size is not None:
            batch_size = self.batch_size
        sess = tf.get_default_session()
        n_samples = len(inputs)
        n_batches = utils.compute_n_batches(n_samples, batch_size)
        probs = np.zeros((len(inputs), self.output_dim))
        for bidx in range(n_batches):
            idxs = utils.compute_batch_idxs(bidx * batch_size, batch_size, n_samples)
            probs[idxs] = sess.run(self.probs, feed_dict={
                self.inputs:inputs[idxs],
                self.dropout_keep_prop_ph: 1.
            })
        preds = np.argmax(probs, axis=-1)
        return probs, preds
    
    def get_param_values(self):
        pass

    def set_params_values(self):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

class FFNN(Network):
    
    def __init__(
            self,
            name,
            input_dim,
            batch_size=100,
            hidden_layer_dims=(128,64),
            **kwargs):
        self.hidden_layer_dims = hidden_layer_dims
        super(FFNN, self).__init__(name, input_dim, is_recurrent=False, **kwargs)
        
    def build_network(self):

        hidden = self.inputs
        for dim in self.hidden_layer_dims:
            hidden = tf.contrib.layers.fully_connected(
                hidden,
                dim,
                activation_fn=tf.nn.relu
            )
            hidden = tf.nn.dropout(hidden, self.dropout_keep_prop_ph)
        self.scores = tf.contrib.layers.fully_connected(
            hidden,
            self.output_dim,
            activation_fn=None
        )
        self.probs = tf.nn.softmax(self.scores)

class RNN(Network):
        
    def __init__(
            self,
            name,
            input_dim,
            max_len,
            hidden_dim=64,
            **kwargs):
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        super(RNN, self).__init__(name, input_dim, is_recurrent=True, **kwargs)

    def build_network(self):
        self.cell = rnn_utils._build_recurrent_cell(self.hidden_dim, self.dropout_keep_prop_ph)
        
        outputs, states = tf.nn.dynamic_rnn(
            self.cell,
            inputs=self.inputs,
            dtype=tf.float32,
            time_major=False
        )

        self.scores = tf.contrib.layers.fully_connected(
            self.cell.get_output(states),
            self.output_dim,
            activation_fn=None
        )
        self.probs = tf.nn.softmax(self.scores)


import numpy as np
import sys
import tensorflow as tf

import utils

class Network(object):
    
    def get_param_values(self):
        pass

    def set_params_values(self):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

class RNN(Network):
    
    def __init__(
            self,
            name,
            input_dim,
            hidden_dim,
            max_len,
            output_dim,
            batch_size=100,
            dropout_keep_prob=1.,
            learning_rate=0.001,
            grad_clip=1.):
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        
        self.build_model()
        
    def build_model(self):
        with tf.variable_scope(self.name):
            self.build_placeholders()
            self.build_network()
            self.build_loss()
            self.build_train_op()
            self.build_summaries()
    
    def build_placeholders(self):
        self.inputs = tf.placeholder(tf.float32, (self.batch_size, self.max_len, self.input_dim), 'inputs')
        self.targets = tf.placeholder(tf.int32, (self.batch_size, self.max_len), 'targets')
        self.lengths = tf.placeholder(tf.int32, (self.batch_size,), 'lengths')
        self.sequence_mask = tf.sequence_mask(self.lengths, maxlen=self.max_len, dtype=tf.float32)
        self.dropout_keep_prop_ph = tf.placeholder_with_default(self.dropout_keep_prob, (), 'dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
    def build_network(self):
        self.cell_fw = utils._build_recurrent_cell(self.hidden_dim, self.dropout_keep_prop_ph)
        self.cell_bw = utils._build_recurrent_cell(self.hidden_dim, self.dropout_keep_prop_ph)
        
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            self.cell_fw,
            self.cell_bw,
            inputs=self.inputs,
            sequence_length=self.lengths,
            dtype=tf.float32,
            time_major=False
        )
        
        outputs = tf.concat(outputs, axis=1)
        outputs = tf.reshape(outputs, (self.batch_size * self.max_len, -1))
        scores = tf.contrib.layers.fully_connected(
            outputs,
            self.output_dim,
            activation_fn=None
        )
        scores = tf.reshape(scores, (self.batch_size, self.max_len, self.output_dim))
        self.scores = scores * tf.expand_dims(self.sequence_mask, 2)
        self.probs = tf.nn.softmax(self.scores) * tf.expand_dims(self.sequence_mask, 2)

        # accuracy
        same = tf.cast(tf.equal(tf.cast(np.argmax(self.scores, axis=-1), tf.int32), self.targets), tf.float32) * self.sequence_mask
        self.acc = tf.reduce_sum(same) / tf.reduce_sum(tf.cast(self.lengths, tf.float32))

        # summaries
        tf.summary.scalar('accuracy', self.acc)
        
    def build_loss(self):
        # compute for each timestep
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.scores)
        # mask, average over timesteps
        loss = tf.reduce_sum(losses * self.sequence_mask, axis=1) / tf.cast(self.lengths, tf.float32)
        # average over batch
        self.loss = tf.reduce_mean(loss)

        # summaries
        tf.summary.scalar('loss', self.loss)
        
    def build_train_op(self):
        self.var_list = tf.trainable_variables()
        
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
            writer=None, 
            val_writer=None):
        sess = tf.get_default_session()
        
        n_batches = utils.compute_n_batches(len(data['train_x']), self.batch_size)
        n_val_batches = utils.compute_n_batches(len(data['val_x']), self.batch_size)
        
        for epoch in range(n_epochs):

            # train
            total_loss = 0
            for bidx in range(n_batches):
                s = bidx * self.batch_size
                e = s + self.batch_size
                
                feed_dict = {
                    self.inputs:data['train_x'][s:e],
                    self.targets:data['train_y'][s:e],
                    self.lengths:data['train_lengths'][s:e]
                }
                outputs_list = [self.loss, self.summary_op, self.global_step, self.train_op]
                loss, summary, step, _ = sess.run(outputs_list, feed_dict=feed_dict)
                total_loss += loss
                writer.add_summary(summary, step)
                sys.stdout.write('\repoch: {} / {} batch: {} / {} loss: {}'.format(
                    epoch+1, n_epochs, bidx+1, n_batches, 
                    total_loss / (self.batch_size * (bidx+1))))
            print('\n')

            # val
            total_loss = 0
            for bidx in range(n_val_batches):
                s = bidx * self.batch_size
                e = s + self.batch_size
                feed_dict = {
                    self.inputs:data['val_x'][s:e],
                    self.targets:data['val_y'][s:e],
                    self.lengths:data['val_lengths'][s:e],
                    self.dropout_keep_prop_ph: 1.
                }
                outputs_list = [self.loss, self.summary_op, self.global_step]
                loss, summary, step = sess.run(outputs_list, feed_dict=feed_dict)
                total_loss += loss
                val_writer.add_summary(summary, step)
                sys.stdout.write('\rval epoch: {} / {} batch: {} / {} loss: {}'.format(
                    epoch+1, n_epochs, bidx+1, n_batches, 
                    total_loss / (self.batch_size * (bidx+1))))
            print('\n')

    def predict(self, inputs, lengths):
        sess = tf.get_default_session()
        n_batches = utils.compute_n_batches(len(inputs), self.batch_size)
        probs = np.zeros((len(inputs), self.max_len, self.output_dim))
        for bidx in range(n_batches):
            s = bidx * self.batch_size
            e = s + self.batch_size

            probs[s:e] = sess.run(self.probs, feed_dict={
                self.inputs:inputs[s:e],
                self.lengths:lengths[s:e],
                self.dropout_keep_prop_ph: 1.
            })

        preds = np.argmax(probs, axis=-1)
        return probs, preds
    
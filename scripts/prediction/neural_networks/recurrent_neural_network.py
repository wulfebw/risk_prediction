"""
A recurrent neural network class
"""
import collections
import numpy as np
import os
import tensorflow as tf
import time

from . import initializers
from .feed_forward_neural_network import NeuralNetwork

class RecurrentNeuralNetwork(NeuralNetwork):
    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        super(RecurrentNeuralNetwork, self).__init__(session, flags)

    def _build_placeholders(self):
        """
        Description:
            - build placeholders for inputs to the tf graph.

        Returns:
            - input_ph: placeholder for a input batch
            - target_ph: placeholder for a target batch
            - dropout_ph: placeholder for fraction of activations 
                to drop
        """
        input_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.timesteps, self.flags.input_dim),
                name="input_ph")
        target_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.output_dim),
                name="target_ph")
        weights_ph = tf.placeholder(tf.float32,
                shape=(None, 1),
                name="weights_ph")
        dropout_ph = tf.placeholder(tf.float32,
                shape=(),
                name="dropout_ph")
        learning_rate_ph = tf.placeholder(tf.float32, 
                shape=(),
                name="lr_ph")

        # summaries
        tf.summary.scalar('dropout keep prob', dropout_ph)
        tf.summary.scalar('learning_rate', learning_rate_ph)
        if self.flags.use_likelihood_weights:
            tf.summary.scalar('weights', tf.reduce_mean(weights_ph))

        return input_ph, target_ph, weights_ph, dropout_ph, learning_rate_ph

    def _build_network(self, input_ph, dropout_ph):
        """
        Description:
            - Builds a recurrent neural network where the features are first
                mapped from the input dim to the hidden dim of the RNN by a 
                feed forward network.

        Args:
            - input_ph: placeholder for the inputs
                shape = (batch_size, input_dim)
            - dropout_ph: placeholder for dropout value

        Returns:
            - scores: the scores for the target values
        """

        # build initializers specific to relu
        weights_initializer = initializers.get_weight_initializer(
            'relu')
        bias_initializer = initializers.get_bias_initializer(
            'relu')

        # build regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(
            self.flags.l2_reg)

        # build hidden layers for feed forward network 
        # if layer dims not set individually, then assume all the same dim
        hidden_layer_dims = self.flags.hidden_layer_dims
        if len(hidden_layer_dims) == 0:
            hidden_layer_dims = [self.flags.hidden_dim 
                for _ in range(self.flags.num_hidden_layers)]

        hidden = input_ph
        for (lidx, hidden_dim) in enumerate(hidden_layer_dims):
            hidden = tf.contrib.layers.fully_connected(hidden, 
                hidden_dim, 
                activation_fn=tf.nn.relu,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=bias_initializer)
            # tf.histogram_summary("layer_{}_activation".format(lidx), hidden)
            if self.flags.use_batch_norm:
                hidden = tf.contrib.layers.batch_norm(hidden)
            hidden = tf.nn.dropout(hidden, dropout_ph)

        # build recurrent network 
        cell = tf.contrib.rnn.GRUCell(num_units=hidden_layer_dims[-1])
        outputs, states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=hidden)

        # build output layer
        last_output = tf.squeeze(tf.slice(
            outputs, (0, self.flags.timesteps - 1, 0), (-1,-1,-1)), 1)
        if self.flags.use_batch_norm:
            last_output = tf.contrib.layers.batch_norm(last_output)
        if self.flags.num_target_bins is not None:
            output_dim = self.flags.output_dim * self.flags.num_target_bins
        else:
            output_dim = self.flags.output_dim
        scores = tf.contrib.layers.fully_connected(last_output, 
                output_dim, 
                activation_fn=None,
                weights_regularizer=weights_regularizer)

        return scores


class ClassificationRecurrentNeuralNetwork(RecurrentNeuralNetwork):
    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        super(ClassificationRecurrentNeuralNetwork, self).__init__(session, flags)

    def predict(self, inputs):
        """
        Description:
            - Predict output values for a set of inputs.

        Args:
            - inputs: input values to predict
                shape = (?, timestpes, input_dim)

        Returns:
            - returns class predictions for each target for each sample.
        """
        # allocate containers
        num_samples = len(inputs)
        pred_y = np.empty((num_samples, self.flags.output_dim))
        pred_probs = np.empty((num_samples, self.flags.output_dim, 
            self.flags.num_target_bins))

        # execute the prediction in batches
        num_batches = int(num_samples / self.flags.batch_size)
        if num_batches * self.flags.batch_size < num_samples:
            num_batches += 1
        for bidx in range(num_batches):
            s = bidx * self.flags.batch_size
            e = s + self.flags.batch_size
            batch = inputs[s:e]
            probs = self.session.run(
                self._probs, feed_dict={self._input_ph: inputs[s:e],
                self._dropout_ph: 1.})
            pred_y[s:e, :] = np.argmax(probs, axis=-1)
            pred_probs[s:e, :, :] = probs
        return pred_y, pred_probs

    def _build_placeholders(self):
        """
        Description:
            - build placeholders for inputs to the tf graph.

        Returns:
            - input_ph: placeholder for a input batch
            - target_ph: placeholder for a target batch
            - dropout_ph: placeholder for fraction of activations 
                to drop
            - learning_rate_ph: placeholder for learning rate
        """
        input_ph, _, weights_ph, dropout_ph, learning_rate_ph = super(
            ClassificationRecurrentNeuralNetwork, self)._build_placeholders()
        target_ph = tf.placeholder(tf.int32,
                shape=(None, self.flags.output_dim),
                name="target_ph")
        return sinput_ph, starget_ph, sweights_ph, sdropout_ph, learning_rate_ph

    def _build_loss(self, scores, targets, weights):
        """
        Description:
            - Build a loss function to optimize using the 
                scores of the network (unnormalized) and 
                the target values

        Args:
            - scores: unnormalized scores output from the network
                shape = (batch_size, output_dim * num_target_bins)
            - targets: the target values
                shape = (batch_size, output_dim)

        Returns:
            - symbolic loss value
        """
        # shape to allow for per-target softmax
        scores = tf.reshape(
            scores, (-1, self.flags.output_dim, self.flags.num_target_bins))

        # per-target softmax
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, 
            labels=targets)

        # probs are the per-target softmax probabilities
        probs = tf.nn.softmax(scores, dim=-1)

        # reduce over target
        loss = tf.reduce_sum(losses, axis=-1)

        if self.flags.use_likelihood_weights:
            loss = loss * weights

        # reduce over batch 
        loss = tf.reduce_sum(loss, axis=0)

        # collect regularization losses
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_loss

        # summaries
        tf.summary.histogram('probs', probs)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('l2 reg loss', reg_loss)

        return loss, probs
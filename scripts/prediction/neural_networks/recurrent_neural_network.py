"""
A recurrent neural network class
"""
import collections
import numpy as np
import os
import tensorflow as tf
import time

from . import models as initializers
from .neural_network_predictor import NeuralNetworkPredictor

class ClassificationRecurrentNeuralNetwork(NeuralNetworkPredictor):
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
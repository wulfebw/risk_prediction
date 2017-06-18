"""
A feed-forward neural network class
"""
import collections
import numpy as np
import os
import tensorflow as tf
import time

from . import models

class NeuralNetwork(object):

    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        self.session = session
        self.flags = flags
        self._build_model()

        # saving and logging setup
        self.saver = tf.train.Saver(
            max_to_keep=100, keep_checkpoint_every_n_hours=.5)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.flags.summary_dir, 'train'), 
            self.session.graph)
        self.test_writer = tf.summary.FileWriter(
            os.path.join(self.flags.summary_dir, 'val'), 
            self.session.graph)
        self.info = collections.defaultdict(list)

    def fit(self, dataset):
        """
        Description:
            - Fit this model to the provided dataset.

        Args:
            - dataset: the dataset to fit. Must implement
                the next_batch function  
        """
        self.start_time = time.time()

        # optionally load
        if self.flags.load_network:
            self.load()

        # fit the model to the dataset over a number of epochs
        for epoch in range(self.flags.num_epochs):
            train_loss, val_loss = 0, 0

            # train epoch
            for bidx, batch in enumerate(dataset.next_batch(validation=False)):
                train_loss += self._run_batch(epoch, bidx, batch, validation=False)
            
            # validation epoch
            for bidx, batch in enumerate(dataset.next_batch(validation=True)):
                val_loss += self._run_batch(epoch, bidx, batch, validation=True)

            # print out progress if verbose
            if self.flags.verbose:
                self.log(epoch, dataset, train_loss, val_loss)

            # snapshot network
            self.save(epoch)

            # update hyperparameters
            self.update()

    def _run_batch(self, epoch, bidx, batch, validation):
        feed_dict = {}

        if self.flags.use_likelihood_weights:
            x, y, w = batch 
            feed_dict[self._weights_ph] = w
        else:
            x, y = batch

        feed_dict[self._input_ph] = x
        feed_dict[self._target_ph] = y
        dropout_keep_prob = 1. if validation else self.flags.dropout_keep_prob
        feed_dict[self._dropout_ph] = dropout_keep_prob
        learning_rate = 0. if validation else self.flags.learning_rate
        feed_dict[self._lr_ph] = learning_rate

        outputs_list = [self._summary_op, self._loss]
        if not validation:
            outputs_list += [self._train_op]
        fetched = self.session.run(outputs_list, feed_dict=feed_dict)

        if validation:
            summary, loss = fetched
        else:
            summary, loss, _ = fetched

        if bidx % self.flags.log_summaries_every == 0:
            writer = self.test_writer if validation else self.train_writer
            writer.add_summary(summary, epoch)

        return loss

    def predict(self, inputs, predict_labels=False):
        """
        Description:
            - Predict output values for a set of inputs.

        Args:
            - inputs: input values to predict
                shape = (?, input_dim)

        Returns:
            - returns probability values for each output.
        """
        num_samples = len(inputs)
        outputs = np.empty((num_samples, self.flags.output_dim))
        if predict_labels:
            pred_y = np.empty((num_samples, self.flags.output_dim))
        num_batches = int(num_samples / self.flags.batch_size)
        if num_batches * self.flags.batch_size < num_samples:
            num_batches += 1
        for bidx in range(num_batches):
            s = bidx * self.flags.batch_size
            e = s + self.flags.batch_size
            batch = inputs[s:e]
            outputs[s:e, :] = self.session.run(
                self._probs, feed_dict={self._input_ph: inputs[s:e],
                self._dropout_ph: 1.})
            if predict_labels:
                pred_y[s:e, :] = np.argmax(outputs[s:e, :], axis=-1).reshape(-1,1)
        if predict_labels:
            return pred_y, outputs
        else:
            return outputs

    def save(self, epoch):
        """
        Description:
            - Save the session and network parameters to checkpoint file.

        Args:
            - epoch: epoch of save
        """
        if epoch % self.flags.save_weights_every == 0:
            if not os.path.exists(self.flags.snapshot_dir):
                os.mkdir(self.flags.snapshot_dir)
            filepath = os.path.join(self.flags.snapshot_dir, 'weights')
            self.saver.save(self.session, filepath, global_step=epoch)

    def load(self):
        """
        Description:
            - Load the lastest checkpoint file if it exists.
        """
        filepath = tf.train.latest_checkpoint(self.flags.snapshot_dir)
        if filepath is not None:
            self.saver.restore(self.session, filepath)

    def log(self, epoch, dataset, train_loss, val_loss):
        """
        Description:
            - Log training information to console

        Args:
            - epoch: training epoch
            - dataset: dataset used for training
            - train_loss: total training loss of the epoch
            - val_loss: total validation loss of the epoch
        """
        self.info['val_loss'].append(val_loss)
        num_train = max(len(dataset.data['x_train']), 1)
        train_loss /= num_train
        num_val = max(len(dataset.data['x_val']), 1)
        val_loss /= num_val
        print('epoch: {}\ttrain loss: {:.6f}\tval loss: {:.6f}\ttime: {:.4f}'.format(
            epoch, train_loss, val_loss, time.time() - self.start_time))

    def update(self):

        # require at least 5 validation losses before computing the decrease
        if len(self.info['val_loss']) > 5:
            # if precentage decrease in validation loss is below a threshold
            # then reduce the learning rate
            past_loss = np.mean(self.info['val_loss'][-5:-1])
            cur_loss = self.info['val_loss'][-1]
            decrease = (past_loss - cur_loss) / past_loss
            if decrease < self.flags.decrease_lr_threshold:
                self.flags.learning_rate *= self.flags.decay_lr_ratio
                self.flags.learning_rate = max(
                    self.flags.learning_rate, self.flags.min_lr)

    def _build_model(self):
        """
        Description:
            - Builds the model, which entails defining placeholders, 
                a network, loss function, and train op. 

                Class variables created during this call all are assigned 
                to self in the body of this function, so everything that is 
                stored should be apparent from looking at this function.

                The results of these methods are passed in / out explicitly.
        """
        # placeholders
        (self._input_ph, self._target_ph, self._weights_ph, self._dropout_ph,
            self._lr_ph) = self._build_placeholders()

        # network
        self._scores = self._build_network(
            self._input_ph, self._dropout_ph)

        # loss
        self._loss, self._probs = self._build_loss(
            self._scores, self._target_ph, self._weights_ph)

        # train operation
        self._train_op = self._build_train_op(self._loss, self._lr_ph)

        # summaries
        self._summary_op = tf.summary.merge_all()

        # intialize the model
        self.session.run(tf.global_variables_initializer())

    def _build_network(self, input_ph, dropout_ph):
        """
        Description:
            - Builds a feed forward network with relu units.

        Args:
            - input_ph: placeholder for the inputs
                shape = (batch_size, input_dim)
            - dropout_ph: placeholder for dropout value

        Returns:
            - scores: the scores for the target values
        """
        return models.build_feed_forward_network(input_ph, dropout_ph, self.flags)

    def _build_loss(self, scores, targets, weights):
        """
        Description:
            - Build a loss function to optimize using the 
                scores of the network (unnormalized) and 
                the target values

        Args:
            - scores: unnormalized scores output from the network
                shape = (batch_size, output_dim)
            - targets: the target values
                shape = (batch_size, output_dim)

        Returns:
            - symbolic loss value
        """

        # create op for probability to use in 'predict'
        probs = tf.sigmoid(scores)

        # create losses, delaying reduction until later
        if self.flags.loss_type == 'ce':
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=scores, labels=targets)
        elif self.flags.loss_type == 'mse':
            losses = (scores - targets) ** 2
        elif self.flags.loss_type == 'mse_probs':
            losses = (probs - targets) ** 2
        elif self.flags.loss_type == 'mse_log_probs':
            targets = tf.clip_by_value(targets, self.flags.eps, 1.)
            targets = tf.log(targets)
            losses = (scores - targets) ** 2
            probs = tf.exp(scores)
        else:
            raise(ValueError("invalid loss type: {}".format(
                self.flags.loss_type)))

        # multiply in weights
        if self.flags.use_likelihood_weights:
            losses = losses * weights

        # reduce accross batch 
        losses = tf.reduce_sum(losses, axis=0)

        # summarize losses individually
        for tidx, target_loss in enumerate(tf.unstack(losses)):
                tf.summary.scalar('target_{}_loss'.format(tidx), target_loss) 

        # overall loss is sum of individual target losses
        loss = tf.reduce_sum(losses)

        # collect regularization losses
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_loss

        # summaries
        tf.summary.histogram('probs', probs)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('l2_reg_loss', reg_loss)

        return loss, probs

    def _build_train_op(self, loss, learning_rate):
        """
        Description:
            - Build a training operation minimizing the loss

        Args:
            - loss: symbolic loss

        Returns:
            - training operation
        """
        # adaptive learning rate
        opt = tf.train.AdamOptimizer(learning_rate)   

        # clip gradients by norm
        grads_params = opt.compute_gradients(loss) 
        clipped_grads_params = [(tf.clip_by_norm(
            g, self.flags.max_norm), p) 
            for (g, p) in grads_params]
        global_step = tf.Variable(0, trainable=False)
        train_op = opt.apply_gradients(
            clipped_grads_params, global_step=global_step)  

        # summaries
        # for (g, p) in clipped_grads_params:
        #     tf.histogram_summary('grads for {}'.format(p.name), g)

        return train_op

class FeedForwardNeuralNetwork(NeuralNetwork):
    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        super(FeedForwardNeuralNetwork, self).__init__(session, flags)

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
                shape=(None, self.flags.input_dim),
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
        tf.summary.scalar('dropout_keep_prob',dropout_ph)
        tf.summary.scalar('learning_rate',learning_rate_ph)
        if self.flags.use_likelihood_weights:
            tf.summary.scalar('weights', tf.reduce_mean(weights_ph))

        return (input_ph, target_ph, weights_ph, dropout_ph, learning_rate_ph)

class ClassificationFeedForwardNeuralNetwork(FeedForwardNeuralNetwork):
    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        super(ClassificationFeedForwardNeuralNetwork, self).__init__(
            session, flags)

    def predict(self, inputs, predict_labels=True):
        """
        Description:
            - Predict output values for a set of inputs.

        Args:
            - inputs: input values to predict
                shape = (?, input_dim)

        Returns:
            - returns class predictions for each target for each sample.
        """
        num_samples = len(inputs)
        pred_y = np.empty((num_samples, self.flags.output_dim))
        pred_probs = np.empty((num_samples, self.flags.output_dim, 
            self.flags.num_target_bins))
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
        """
        input_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.input_dim),
                name="input_ph")
        target_ph = tf.placeholder(tf.int32,
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
        tf.summary.scalar('weights', tf.reduce_mean(weights_ph))

        return input_ph, target_ph, weights_ph, dropout_ph, learning_rate_ph

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
        
        # # summarize losses individually
        # for tidx, target_loss in enumerate(tf.unstack(losses, axis=-1)):
        #     print(target_loss.get_shape())
        #     input()
        #     tf.summary.scalar('target_{}_loss'.format(tidx), target_loss) 

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

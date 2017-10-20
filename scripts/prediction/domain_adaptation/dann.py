
import numpy as np
import sys
import tensorflow as tf

from models import encoder, classifier
from flip_gradient import flip_gradient
from utils import compute_n_batches, compute_batch_idxs

class DANN(object):
    
    def __init__(
            self,
            input_dim,
            output_dim,
            name='dann',
            learning_rate=0.0002,
            lambda_initial=0.,
            lambda_final=1.,
            lambda_steps=1000,
            dropout_keep_prob=1.,
            encoder_hidden_layer_dims=(64,64),
            domain_classifier_hidden_layer_dims=(64,64)):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.learning_rate = learning_rate
        self.lambda_initial = lambda_initial
        self.lambda_final = lambda_final
        self.lambda_steps = lambda_steps
        self.dropout_keep_prob = dropout_keep_prob
        self.encoder_hidden_layer_dims = encoder_hidden_layer_dims
        self.domain_classifier_hidden_layer_dims = domain_classifier_hidden_layer_dims
        
        self.build_model()
    
    def build_model(self):
        with tf.variable_scope(self.name):
            self.build_placeholders()
            self.build_networks()
            self.build_loss()
            self.build_train_op()

    def build_placeholders(self):
        self.x = tf.placeholder(tf.float32, (None, self.input_dim), 'x')
        self.y = tf.placeholder(tf.float32, (None, self.output_dim), 'y')
        self.d = tf.placeholder(tf.int32, (None), 'domain')
        self.dropout_keep_prob_ph = tf.placeholder_with_default(
            self.dropout_keep_prob, (), 'dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
    
    def build_networks(self):
        # feature extractor
        self.features = encoder(
            self.x, 
            'features', 
            hidden_layer_dims=self.encoder_hidden_layer_dims,
            dropout_keep_prob=self.dropout_keep_prob_ph
        )
        
        # task classifier
        self.task_scores = classifier(
            self.features, 
            self.output_dim,
            'task_classifier',
        )
        self.task_probs = tf.nn.softmax(self.task_scores)
        pred = tf.cast(tf.argmax(self.task_scores, axis=-1), tf.float32)
        true = tf.cast(tf.argmax(self.y, axis=-1), tf.float32)
        self.task_acc = tf.reduce_mean(tf.cast(tf.equal(true, pred), tf.float32))
        
        # domain classifier
        lmbda = tf.train.polynomial_decay(
            self.lambda_initial, 
            self.global_step, 
            self.lambda_steps, 
            end_learning_rate=self.lambda_final, 
            power=2.0,
            name='lambda'
        )
        flipped_features = flip_gradient(self.features, lmbda)
        domain_features = encoder(
            flipped_features,
            'domain_features',
            hidden_layer_dims=self.domain_classifier_hidden_layer_dims,
            dropout_keep_prob=self.dropout_keep_prob_ph
        )
        self.domain_scores = classifier(
            domain_features, 
            2,
            'domain_classifier'
        )
        self.domain_probs = tf.nn.softmax(self.domain_scores)
        
        pred = tf.cast(tf.argmax(self.domain_scores, axis=-1), tf.int32)
        self.domain_acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.d), tf.float32))
       
    def build_loss(self):
        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.task_scores
        ))
        self.domain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.d, logits=self.domain_scores
        ))
        self.loss = self.task_loss + self.domain_loss
        
    def build_train_op(self):
        # for comparing DANN to regular mixed training create a training only operation
        self.task_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.task_loss, 
            global_step=self.global_step
        )
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss, 
            global_step=self.global_step
        )
        
    def train_batch(self, batch, epoch, n_epochs, train=True, writer=None):
        sess = tf.get_default_session()
        outputs_list = [self.loss, self.domain_acc]
        outputs_list += [self.train_op] if train else []
        feed = {
            self.x: batch['x'],
            self.y: batch['y'],
            self.d: batch['d'],
            self.dropout_keep_prob_ph: self.dropout_keep_prob if train else 1.
        }
        fetched = sess.run(outputs_list, feed_dict=feed)
        if train:
            loss, acc, _ = fetched
        else:
            loss, acc = fetched
        sys.stdout.write('\r training: {} epoch: {} / {} loss: {:.4f} domain accuracy: {:.4f}'.format(
                train, epoch+1, n_epochs, loss, acc))
        
        return loss, acc
        
    def train(
            self,
            dataset,
            val_dataset=None,
            n_epochs=100,
            writer=None,
            val_writer=None):
        
        sess = tf.get_default_session()
    
        for epoch in range(n_epochs):
            
            for batch in dataset.batches():
                self.train_batch(batch, epoch, n_epochs, train=True, writer=writer)
            if val_dataset is not None:
                for batch in val_dataset.batches():
                    self.train_batch(batch, epoch, n_epochs, train=False, writer=val_writer)
                    
    def evaluate(self, x, y, d, batch_size=100):
        sess = tf.get_default_session()
        n_samples = len(x)
        n_batches = compute_n_batches(n_samples, batch_size)
        probs = np.zeros((len(x), self.output_dim))
        outputs_list = [
            self.task_probs, 
            self.task_loss, 
            self.task_acc,
            self.domain_acc
        ]
        task_loss_list = []
        task_acc_list = []
        domain_acc_list = []
        for bidx in range(n_batches):
            idxs = compute_batch_idxs(bidx * batch_size, batch_size, n_samples)
            cur_probs, task_loss, task_acc, domain_acc = sess.run(outputs_list, feed_dict={
                self.x: x[idxs],
                self.y: y[idxs],
                self.d: d[idxs],
                self.dropout_keep_prob_ph: 1.
            })
            probs[idxs] = cur_probs
            task_loss_list.append(task_loss)
            task_acc_list.append(task_acc)
            domain_acc_list.append(domain_acc)
        return dict(
            probs=probs, 
            task_loss=np.mean(task_loss_list), 
            task_acc=np.mean(task_acc_list),
            domain_acc=np.mean(domain_acc_list)
        )
        
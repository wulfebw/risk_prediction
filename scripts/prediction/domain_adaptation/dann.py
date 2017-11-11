
from collections import defaultdict
import numpy as np
import sys
import tensorflow as tf

from models import encoder, classifier
from flip_gradient import flip_gradient
from utils import compute_n_batches, compute_batch_idxs, classification_score, evaluate, process_stats

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
            classifier_hidden_layer_dims=(),
            domain_classifier_hidden_layer_dims=(64,64),
            src_only_adversarial=False,
            shared_classifier=True,
            da_mode='unsupervised'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.learning_rate = learning_rate
        self.lambda_initial = lambda_initial
        self.lambda_final = lambda_final
        self.lambda_steps = lambda_steps
        self.dropout_keep_prob = dropout_keep_prob
        self.encoder_hidden_layer_dims = encoder_hidden_layer_dims
        self.classifier_hidden_layer_dims = classifier_hidden_layer_dims
        self.domain_classifier_hidden_layer_dims = domain_classifier_hidden_layer_dims
        self.src_only_adversarial = src_only_adversarial
        self.shared_classifier = shared_classifier
        self.da_mode = da_mode
        
        self.build_model()
    
    def build_model(self):
        with tf.variable_scope(self.name):
            self.build_placeholders()
            self.build_networks()
            self.build_loss()
            self.build_train_op()

    def build_placeholders(self):
        self.xs = tf.placeholder(tf.float32, (None, self.input_dim), 'xs')
        self.xt = tf.placeholder(tf.float32, (None, self.input_dim), 'xt')
        self.ws = tf.placeholder(tf.float32, (None,), 'ws')
        self.ys = tf.placeholder(tf.float32, (None, self.output_dim), 'ys')
        self.yt = tf.placeholder(tf.float32, (None, self.output_dim), 'yt')
        self.wt = tf.placeholder(tf.float32, (None,), 'wt')
        
        self.dropout_keep_prob_ph = tf.placeholder_with_default(
            np.float32(self.dropout_keep_prob), (), 'dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
    
    def build_networks(self):

        # feature extractor
        self.src_features = src_features = encoder(
            self.xs, 
            'features', 
            hidden_layer_dims=self.encoder_hidden_layer_dims,
            dropout_keep_prob=self.dropout_keep_prob_ph
        )

        self.tgt_features = tgt_features = encoder(
            self.xt, 
            'features', 
            hidden_layer_dims=self.encoder_hidden_layer_dims,
            dropout_keep_prob=self.dropout_keep_prob_ph,
            reuse=True
        )

        # task src classifier
        src_classifier_feat = encoder(
            src_features,
            'src_classifier_feat',
            hidden_layer_dims=self.classifier_hidden_layer_dims,
            dropout_keep_prob=self.dropout_keep_prob_ph
        )
        self.src_task_scores = classifier(
            src_classifier_feat, 
            self.output_dim,
            'src_task_scores',
        )
        self.src_task_probs = tf.nn.softmax(self.src_task_scores)

        # task tgt classifier with optional reuse
        if self.shared_classifier:
            reuse = True
            tgt_classifier_feat_name = 'src_classifier_feat'
            tgt_classifier_name = 'src_task_scores'
        else:
            reuse = False
            tgt_classifier_feat_name = 'tgt_classifier_feat'
            tgt_classifier_name = 'tgt_task_scores'

        tgt_classifier_feat = encoder(
            tgt_features,
            tgt_classifier_feat_name,
            hidden_layer_dims=self.classifier_hidden_layer_dims,
            dropout_keep_prob=self.dropout_keep_prob_ph,
            reuse=reuse
        )
        self.tgt_task_scores = classifier(
            tgt_classifier_feat, 
            self.output_dim,
            tgt_classifier_name,
            reuse=reuse
        )
        self.tgt_task_probs = tf.nn.softmax(self.tgt_task_scores)

        # domain classifier
        self.lmbda = lmbda = tf.train.polynomial_decay(
            self.lambda_initial, 
            self.global_step, 
            self.lambda_steps, 
            end_learning_rate=self.lambda_final, 
            power=2.0,
            name='lambda'
        )

        if self.src_only_adversarial:
            tgt_features = tf.stop_gradient(tgt_features)
        self.features = tf.concat((src_features, tgt_features), axis=0)
        flipped_features = flip_gradient(self.features, lmbda)
        src_d = tf.zeros((tf.shape(src_features)[0],), dtype=tf.int32)
        tgt_d = tf.ones((tf.shape(tgt_features)[0],), dtype=tf.int32)
        self.d = tf.concat((src_d, tgt_d), axis=0)

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
        # task
        self.src_task_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.ys, logits=self.src_task_scores
            ) * self.ws
        )
        self.tgt_task_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.yt, logits=self.tgt_task_scores
            ) * self.wt
        )
        self.task_loss = self.src_task_loss + self.tgt_task_loss

        # domain
        if self.da_mode == 'unsupervised':
            self.domain_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.d, logits=self.domain_scores
                ) * tf.concat((self.ws, self.wt), axis=0)
            )
        elif self.da_mode == 'supervised':
            feature_l2 = tf.reduce_sum((self.src_features - self.tgt_features) ** 2, axis=1)
            feature_l2 = tf.maximum(feature_l2, 1e-8)
            feature_l2 = tf.sqrt(feature_l2)
                    
            w = (1 - (self.ys[:,1] - self.yt[:,1]) ** 2) * self.ws * self.wt
            self.domain_loss = tf.reduce_mean(feature_l2 * w) * self.lmbda

        # overall
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
        outputs_list = [self.src_task_loss, self.tgt_task_loss, self.tgt_task_probs, self.domain_acc]
        outputs_list += [self.train_op] if train else []
        feed = {
            self.xs: batch['xs'],
            self.ys: batch['ys'],
            self.ws: batch['ws'],
            self.xt: batch['xt'],
            self.yt: batch['yt'],
            self.wt: batch['wt'],
            self.dropout_keep_prob_ph: self.dropout_keep_prob if train else 1.
        }
        fetched = sess.run(outputs_list, feed_dict=feed)
        if train:
            src_loss, tgt_loss, tgt_probs, acc, _ = fetched
        else:
            src_loss, tgt_loss, tgt_probs, acc = fetched

        sys.stdout.write(
            '\r training: {} epoch: {} / {} src loss: {:.6f} tgt loss: {:.6f} domain accuracy: {:.4f}'.format(
            train, epoch+1, n_epochs, src_loss, tgt_loss, acc))

        ret = dict(src_loss=src_loss, tgt_loss=tgt_loss)
        if not train:
            ret.update(evaluate(batch['yt'], tgt_probs, batch['wt']))
        return ret
        
    def train(
            self,
            dataset,
            val_dataset=None,
            val_every=1,
            n_epochs=100,
            writer=None,
            val_writer=None,
            verbose=True):
        
        sess = tf.get_default_session()
        stats = defaultdict(lambda: defaultdict(list))
    
        for epoch in range(n_epochs):
            
            for batch in dataset.batches():
                batch_stats = self.train_batch(
                    batch, epoch, n_epochs, train=True, writer=writer)
                stats[epoch]['train'].append(batch_stats)
            if val_dataset is not None and epoch % val_every == 0:
                for batch in val_dataset.batches():
                    batch_stats = self.train_batch(
                        batch, epoch, n_epochs, train=False, writer=val_writer)
                    stats[epoch]['val'].append(batch_stats)

            if verbose:
                self.report(stats, epoch+1)

        return stats

    def report(self, stats, epoch):
        stats = process_stats(dict(stats=stats))

        print('\n')
        print('epoch: {}'.format(epoch))
        print('train src loss: {}'.format(stats['train']['src_loss'][-1]))
        print('train tgt loss: {}'.format(stats['train']['tgt_loss'][-1]))
        if len(stats['val'].keys()) > 0:
            print('val src loss: {}'.format(stats['val']['src_loss'][-1]))
            print('val tgt loss: {}'.format(stats['val']['tgt_loss'][-1]))
            print('val tgt pos loss: {}'.format(stats['val']['pos_ce'][-1]))
            print('val tgt brier: {}'.format(stats['val']['brier'][-1]))
            print('val tgt pos brier: {}'.format(stats['val']['pos_brier'][-1]))
        print('\n')

    def predict(self, x, tgt=True, batch_size=100):
        # setup
        sess = tf.get_default_session()
        n_samples = len(x)
        n_batches = compute_n_batches(n_samples, batch_size)
        probs = np.zeros((n_samples, self.output_dim))

        # decide between src or tgt probs
        outputs_list = [self.tgt_task_probs] if tgt else [self.src_task_probs]
        x_ph = self.xt if tgt else self.xs

        # compute probs
        for bidx in range(n_batches):
            idxs = compute_batch_idxs(bidx * batch_size, batch_size, n_samples, fill='none')
            probs[idxs] = sess.run(outputs_list, feed_dict={
                x_ph: x[idxs],
                self.dropout_keep_prob_ph: 1.
            })

        return probs
        
from __future__ import print_function
from collections import namedtuple
import copy
import logging
import numpy as np
np.set_printoptions(precision=6, suppress=True)
import six.moves.queue as queue
import scipy.signal
import tensorflow as tf
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from . import model
from . import build_envs

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma):
    """
    given a rollout, compute its returns
    """
    batch_si = np.asarray(rollout.states)
    rewards = np.asarray(rollout.rewards)
    batch_w = np.asarray(rollout.weights)
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]

    features = rollout.features[0]
    return Batch(batch_si, batch_r, batch_w, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "r", "w", "terminal", "features"])

class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, value_dim=5):
        self.states = []
        self.rewards = []
        self.weights = []
        self.r = np.zeros(value_dim)
        self.terminal = False
        self.features = []

    def add(self, state, reward, weight, terminal, features):
        self.states += [state]
        self.rewards += [reward]
        self.weights += [weight]
        self.terminal = terminal
        self.features += [features]

def env_runner(env, policy, num_local_steps, summary_writer, value_dim=5, 
        verbose=True, visualize=True, visualize_every=1000):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    total_rewards = np.zeros(value_dim)
    total_length = 0
    ep_count = 0
    rewards = np.zeros(value_dim)
    
    while True:
        terminal_end = False
        rollout = PartialRollout(value_dim=value_dim)

        for local_step in range(num_local_steps):
            features = policy.features(last_state, *last_features)
            state, reward, terminal, info = env.step(None)

            # if len(np.shape(terminal)) > 0:
            #     reward = np.sum(reward, axis=0) / len(terminal)
            #     state = state[-1]
            #     terminal = terminal[-1]
            #     # total_reward = np.zeros_like(reward[0])

                # for i, t in enumerate(terminal[:-1]):
                #     if t:
                #         total_reward += reward[i]
                #     else:
                #         total_reward += policy.value(state[i], *features)

                # if terminal[-1]:
                #     total_reward += reward[-1]
                #     total_reward /= len(terminal)
                # else:
                #     total_reward /= len(terminal[:-1])
                #     likelihood *= 1. / len(terminal)

                # reward = total_reward
                # state = state[-1]
                # terminal = terminal[-1]

            if visualize and ep_count % visualize_every == 0:
                env.render()

            # collect the experience
            # note that the deepcopies seem to be necessary
            rollout.add(
                copy.deepcopy(last_state), 
                copy.deepcopy(reward), 
                info['weight'], 
                terminal, 
                copy.deepcopy(last_features))
            
            length += 1
            rewards += reward
            
            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()


            timestep_limit = env.spec.tags.get(
                'wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                ep_count += 1
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                
                total_rewards += rewards
                total_length += length
                avg_rewards = total_rewards / ep_count
                avg_length = total_length / ep_count
                if verbose:
                    print("Episode finished\tSum of rewards: {}\tLength: {}\tAverage Rewards: {}\tAverage Length: {:.2f}".format(
                        rewards, length, avg_rewards, avg_length))

                length = 0
                rewards = np.zeros(value_dim)
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it
        yield rollout

class AsyncTD(object):
    def __init__(self, env, task, config):
        self.env = env
        self.task = task
        self.config = config

        # when testing only on localhost, use simple worker device
        if config.testing:
            worker_device = "/job:localhost/replica:0/task:0/cpu:0"
            global_device = worker_device
        else:
            worker_device = "/job:worker/task:{}/cpu:0".format(task)
            global_device = tf.train.replica_device_setter(
                1, worker_device=worker_device)

        with tf.device(global_device):
            with tf.variable_scope("global"):
                self.network = model.LSTMPredictor(env.observation_space.shape, config)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = model.LSTMPredictor(
                    env.observation_space.shape, config)
                pi.global_step = self.global_step

            self.r = tf.placeholder(tf.float32, [None, config.value_dim], name="r")
            self.w = tf.placeholder(tf.float32, [None], name='sample_weights')

            # loss of value function
            self.loss = self._build_loss()

            # grads
            grads = tf.gradients(self.loss, pi.var_list)

            # summaries
            ## input summaries
            tf.summary.histogram("model/sample_weights", self.w[0])
            tf.summary.scalar("model/sample_weights", self.w[0])
            if self.config.summarize_features:
                for i, feature_name in enumerate(build_envs.get_obs_var_names(env)):
                    tf.summary.scalar("features/{}_value".format(
                        feature_name.encode('utf-8')), 
                        tf.reduce_mean(pi.x[:,i]))
            ## target and loss summaries
            bs = tf.to_float(tf.shape(pi.x)[0])
            tf.summary.scalar("model/value_loss", self.loss / bs)
            mean_vf = tf.reduce_mean(pi.vf, axis=0)
            if self.config.loss_type == 'ce':
                mean_vf = tf.nn.sigmoid(mean_vf)
            tf.summary.scalar("model/value_mean", tf.reduce_mean(pi.vf))
            for i, target_name in enumerate(
                    build_envs.get_target_names(env, self.config.value_dim)):
                tf.summary.scalar("model/value_mean_{}".format(target_name), 
                    mean_vf[i])
                tf.summary.scalar("model/value_{}".format(target_name), 
                    pi.vf[0,i])
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))

            # loss
            grads, _ = tf.clip_by_global_norm(grads, config.grad_clip_norm)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) 
                for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # learning rate decay
            learning_rate = tf.train.polynomial_decay(
                self.config.learning_rate, 
                self.global_step,
                end_learning_rate=config.learning_rate_end,
                decay_steps=self.config.n_global_steps,
                power=2
            )
            tf.summary.scalar("model/learning_rate", learning_rate)

            # each worker has a different set of adam optimizer parameters
            optimizers = {
                'adam': tf.train.AdamOptimizer(
                    learning_rate, 
                    beta1=config.adam_beta1,
                    beta2=config.adam_beta2,
                    epsilon=config.adam_epsilon
                ),
                'rmsprop': tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=config.rmsprop_decay,
                    momentum=config.rmsprop_momentum
                ),
            }

            opt = optimizers[config.optimizer]
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

            # merge the summaries
            self.summary_op = tf.summary.merge_all()

    def start(self, sess, summary_writer):
        self.rollout_provider = env_runner(self.env, self.local_network, 
            self.config.local_steps_per_update, summary_writer, 
            value_dim=self.config.value_dim, 
            visualize=self.config.visualize and self.task == 0,
            visualize_every=self.config.visualize_every)
        self.summary_writer = summary_writer

    def process(self, sess):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """

        sess.run(self.sync)  # copy weights from shared to local
        rollout = next(self.rollout_provider)
        batch = process_rollout(rollout, gamma=self.config.discount)

        should_compute_summary = (self.task == 0 
            and self.local_steps % self.config.summary_every == 0)

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.r: batch.r,
            self.w: batch.w,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
            self.local_network.dropout_keep_prob_ph: self.config.dropout_keep_prob
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(
                tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

    def validate(self, sess, data):
        """
        Validates the model against a given dataset.
        """
        if data is None:
            return

        # copy weights from shared to local
        sess.run(self.sync)  

        # compute the average rmse between the predicted and true values 
        # across the validation dataset
        total_loss = 0
        total_w = 0
        total_v = 0
        # predict value for each sample in the dataset
        # x.shape = (n_timesteps, input_dim)
        # y.shape = (value_dim,)
        # w is a scalar giving the weight of this sample
        for (x, y, w) in data.next_batch():
            # compute the value
            v = self.local_network.value(x, 
                self.local_network.state_init[0],
                self.local_network.state_init[1],
                sequence=True)

            total_loss += np.sqrt((v - y) ** 2) * w
            total_w += w
            total_v += v * w
        avg_loss = total_loss / total_w
        avg_value = total_v / total_w
        print('Average Validation Loss: {}'.format(avg_loss))

        # write the summary
        summaries = []
        for (i, target_name) in enumerate(
                build_envs.get_target_names(self.env, self.config.value_dim)):
            summaries += [tf.Summary.Value(
                tag="val/{}_validation_loss".format(target_name), 
                simple_value=float(avg_loss[i]))]
            summaries += [tf.Summary.Value(
                tag="val/{}_value".format(target_name), 
                simple_value=float(avg_value[i]))]
        summary = tf.Summary(value=summaries)

        self.summary_writer.add_summary(summary)
        self.summary_writer.flush()
        return avg_loss

    def _build_squared_error_loss_component(self, scores, targets, w, 
            target_loss_index):
        td_error = tf.square(scores - targets)
        if target_loss_index is not None:
            loss = tf.reduce_sum(w * tf.reduce_mean(
                    td_error[:, target_loss_index], axis=-1))
        else:
            loss = tf.reduce_sum(w * tf.reduce_mean(td_error, axis=-1))

        mean_targets = tf.reduce_mean(self.r, axis=0)
        mean_target_td_errors = tf.reduce_mean(td_error, axis=0)

        for i, target_name in enumerate(
                build_envs.get_target_names(self.env, np.shape(targets)[-1])):
            tf.summary.scalar("targets/{}_value".format(target_name), 
                mean_targets[i])
            tf.summary.scalar("targets/{}_loss".format(target_name), 
                mean_target_td_errors[i])

        return loss

    def _my_sigmoid_cross_entropy_with_logits(self, logits, labels, e=1e-8):
        loss = (labels * -tf.log(tf.nn.sigmoid(logits)) 
            + (1 - labels) * -tf.log(1 - tf.nn.sigmoid(logits)))
        return loss

    def _build_cross_entropy_loss_component(self, scores, targets, w, 
            target_loss_index):
        if target_loss_index is not None:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=scores[:, target_loss_index], 
                labels=targets[:, target_loss_index])
            # loss = self._my_sigmoid_cross_entropy_with_logits(
            #     logits=scores[:, target_loss_index], 
            #     labels=targets[:, target_loss_index])
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=scores, labels=targets)
            # loss = self._my_sigmoid_cross_entropy_with_logits(
            #     logits=scores, labels=targets)

        loss = loss * tf.reshape(w, [-1,1])

        mean_targets = tf.reduce_mean(self.r, axis=0)
        mean_target_ce_errors = tf.reduce_mean(loss, axis=0)
        done = False
        for i, target_name in enumerate(
                build_envs.get_target_names(self.env, np.shape(targets)[-1])):
            tf.summary.scalar("targets/{}_value".format(target_name), 
                mean_targets[i])
            if target_loss_index is not None:
                if not done:
                    tf.summary.scalar("targets/{}_loss".format(target_name), 
                        mean_target_ce_errors)
                    done = True
            else:
                tf.summary.scalar("targets/{}_loss".format(target_name), 
                    mean_target_ce_errors[i])

        if target_loss_index is None:
            loss = tf.reduce_mean(loss, axis=-1)
        loss = tf.reduce_sum(loss)
        return loss

    def _build_loss(self):
        pi = self.local_network

        # log mse
        if self.config.loss_type == 'log_mse':
            r = tf.log(tf.clip_by_value(self.r, self.config.eps, 1))
            loss = self._build_squared_error_loss_component(
                pi.vf, self.r, self.w, self.config.target_loss_index)
            
        # cross entropy loss
        elif self.config.loss_type == 'ce':
            loss = self._build_cross_entropy_loss_component(
                pi.vf, self.r, self.w, self.config.target_loss_index)

        # mse / brier
        elif self.config.loss_type == 'mse':
            loss = self._build_squared_error_loss_component(
                pi.vf, self.r, self.w, self.config.target_loss_index)

        # l2 regularization loss
        reg_loss = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.config.l2_reg),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
            tf.get_variable_scope().name))
        loss += reg_loss
        tf.summary.scalar("model/l2_reg_loss", reg_loss)

        return loss

        
        

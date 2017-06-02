from __future__ import print_function
from collections import namedtuple
import copy
import logging
from model import LSTMPredictor
import numpy as np
import six.moves.queue as queue
import scipy.signal
import tensorflow as tf
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import build_envs

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

def env_runner(env, policy, num_local_steps, summary_writer):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = np.zeros(5)
    
    while True:
        terminal_end = False
        rollout = PartialRollout()

        for local_step in range(num_local_steps):
            features = policy.features(last_state, *last_features)
            state, reward, terminal, info = env.step(None)

            if len(np.shape(terminal)) > 0:
                reward = np.sum(reward, axis=0) / len(terminal)
                state = state[-1]
                terminal = terminal[-1]
                # total_reward = np.zeros_like(reward[0])

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
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: {}. Length: {}".format(rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it
        yield rollout

class A3C(object):
    def __init__(self, env, task, config):
        self.env = env
        self.task = task
        self.config = config
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPredictor(env.observation_space.shape, config)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPredictor(
                    env.observation_space.shape, config)
                pi.global_step = self.global_step

            self.r = tf.placeholder(tf.float32, [None, config.value_dim], name="r")
            self.w = tf.placeholder(tf.float32, [None], name='sample_weights')

            # loss of value function
            print('self.w.shape ', self.w.shape)
            print('pi.vf.shape ', pi.vf.shape)
            print('self.r.shape ', self.r.shape)
            td_error = tf.square(pi.vf - self.r)
            if config.target_loss_index is not None:
                self.loss = tf.reduce_sum(self.w * tf.reduce_mean(
                    td_error[:, config.target_loss_index], axis=-1))
            else:
                self.loss = tf.reduce_sum(self.w * tf.reduce_mean(td_error, axis=-1))
            print('self.loss.shape ', self.loss.shape)

            # grads
            grads = tf.gradients(self.loss, pi.var_list)

            # summaries
            ## input summaries
            tf.summary.histogram("model/sample_weights", self.w[0])
            tf.summary.scalar("model/sample_weights", self.w[0])

            julia_env = build_envs.get_julia_env(self.env)
            if self.config.summarize_features:
                for i, feature_name in enumerate(julia_env.obs_var_names()):
                    tf.summary.scalar("features/{}_value".format(
                        feature_name.encode('utf-8')), 
                        tf.reduce_mean(pi.x[:,i]))

            ## target and loss summaries
            mean_vf = tf.reduce_mean(pi.vf, axis=0)
            tf.summary.scalar("model/value_mean", tf.reduce_mean(pi.vf))
            for i, target_name in enumerate(julia_env.reward_names()):
                tf.summary.scalar("model/value_mean_{}".format(target_name), 
                    mean_vf[i])
            bs = tf.to_float(tf.shape(pi.x)[0])
            tf.summary.scalar("model/value_loss", self.loss / bs)
            mean_targets = tf.reduce_mean(self.r, axis=0)
            mean_target_td_errors = tf.reduce_mean(td_error, axis=0)
            for i, target_name in enumerate(julia_env.reward_names()):
                tf.summary.scalar("targets/{}_value".format(target_name), 
                    mean_targets[i])
                tf.summary.scalar("targets/{}_loss".format(target_name), 
                    mean_target_td_errors[i])

            ## gradient and variable norm
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))

            # merge the summaries
            self.summary_op = tf.summary.merge_all()

            # loss
            grads, _ = tf.clip_by_global_norm(grads, config.grad_clip_norm)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) 
                for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            if config.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(config.learning_rate)
            elif config.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(config.learning_rate, momentum=.9)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.rollout_provider = env_runner(self.env, self.local_network, 
            self.config.local_steps_per_update, summary_writer)
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
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(
                tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

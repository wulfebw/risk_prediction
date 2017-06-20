
import gym
import numpy as np
import os
import sys
import tensorflow as tf
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(os.path.abspath(path))
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

from prediction import async_td
from prediction import build_envs
from prediction import validation
from test_config import TestConfig
from envs import debug_envs

class TestAsyncTD(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()

    def test_prediction_single_step_hard_brake(self):
        config = TestConfig()
        config.n_global_steps = 50000
        config.env_id = 'HeuristicRiskEnv-v0'
        config.discount = 0.
        config.learning_rate = 5e-5
        config.adam_beta1 = .99
        config.adam_beta2 = .999
        config.dropout_keep_prob = 1.
        config.l2_reg = 0.
        config.hidden_layer_sizes = [16, 8]
        config.hard_brake_threshold = 0.
        config.local_steps_per_update = 1
        config.hard_brake_n_past_frames = 1
        config.target_loss_index = 3
        config.loss_type = 'ce'
        config.normalization_type = 'range'
        env = build_envs.create_env(config)
        test_state = env.reset()
        summary_writer = tf.summary.FileWriter('/tmp/test')
        with tf.Session() as sess:
            trainer = async_td.AsyncTD(env, 0, config)
            sess.run(tf.global_variables_initializer())
            sess.run(trainer.sync)
            trainer.start(sess, summary_writer)
            global_step = sess.run(trainer.global_step)
            c, h = trainer.network.get_initial_features()
            while global_step < config.n_global_steps:
                trainer.process(sess)
                global_step = sess.run(trainer.global_step)
                value = trainer.network.value(test_state, c, h)
                print(value)
        self.assertTrue(value[3] < .55 and value[3] > .45)

    def test_prediction_long_term_hard_brake(self):
        config = TestConfig()
        config.n_global_steps = 20000
        config.env_id = 'HeuristicRiskEnv-v0'
        config.discount = 0. # 599. / 600
        config.max_timesteps = 10000
        config.prime_timesteps = 50
        config.learning_rate = 1e-3
        config.adam_beta1 = .995
        config.adam_beta2 = .999
        config.dropout_keep_prob = 1.
        config.l2_reg = 0.
        config.local_steps_per_update = 20
        config.hidden_layer_sizes = [32, 16]
        config.hard_brake_threshold = -3.
        config.hard_brake_n_past_frames = 1
        config.target_loss_index = 3
        config.loss_type = 'mse'
        env = build_envs.create_env(config)
        test_state = env.reset()
        summary_writer = tf.summary.FileWriter('/tmp/test')
        with tf.Session() as sess:
            trainer = async_td.AsyncTD(env, 0, config)
            sess.run(tf.global_variables_initializer())
            sess.run(trainer.sync)
            trainer.start(sess, summary_writer)
            global_step = sess.run(trainer.global_step)
            c, h = trainer.network.get_initial_features()
            while global_step < config.n_global_steps:
                trainer.process(sess)
                global_step = sess.run(trainer.global_step)
                value = trainer.network.value(test_state, c, h)
                print(value)
            # self.assertTrue(value[3] > .5 and value[3] < .6)
    
    def test_validate_const_reward_discounted_env(self):
        # config
        config = TestConfig()
        config.n_global_steps = 50000
        config.env_id = 'RandObsConstRewardEnv-v0'
        config.discount = .9
        config.value_dim = 2
        config.adam_beta1 = .9
        config.local_steps_per_update = 1000
        config.hidden_layer_sizes = [256]
        config.learning_rate = 1e-3
        config.learning_rate_end = 1e-5
        config.loss_type = 'mse'
        config.target_loss_index = None

        # build env
        const_reward = .01
        horizon = 10000000
        rand_obs = False
        env = debug_envs.RandObsConstRewardEnv(
            horizon=horizon, 
            reward=const_reward, 
            value_dim=config.value_dim,
            rand_obs=rand_obs
        )
        env.spec = gym.envs.registration.EnvSpec(
            id='RandObsConstRewardEnv-v0', 
            tags={'wrapper_config.TimeLimit.max_episode_steps': horizon+1}
        )
        
        n_samples = 2
        n_timesteps = 10 # predict after seeing this many timesteps
        n_prediction_timesteps = 10 # determines discount
        input_dim = 1
        obs_gen = np.random.randn if rand_obs else np.ones
        x = obs_gen(np.prod((n_samples, n_timesteps, input_dim))).reshape(
            (n_samples, n_timesteps, input_dim))
        y = (const_reward * np.ones((n_samples, config.value_dim)) 
            * n_prediction_timesteps)
        w = np.ones((n_samples,1))
        dataset = validation.Dataset(x, y, w)

        # run it
        summary_writer = tf.summary.FileWriter('/tmp/test')
        avg_loss = -1
        with tf.Session() as sess:
            trainer = async_td.AsyncTD(env, 0, config)
            sess.run(tf.global_variables_initializer())
            sess.run(trainer.sync)
            trainer.start(sess, summary_writer)
            global_step = sess.run(trainer.global_step)
            while global_step < config.n_global_steps:
                trainer.process(sess)
                if global_step % 10 == 0:
                    avg_loss = trainer.validate(sess, dataset)
                global_step = sess.run(trainer.global_step)


    def test_validate(self):
        # config
        config = TestConfig()
        config.n_global_steps = 5000
        config.env_id = 'SeqSumDebugEnv-v0'
        config.discount = 1.
        config.value_dim = 1
        config.adam_beta1 = .99
        config.local_steps_per_update = 100000
        config.hidden_layer_sizes = [128]
        config.learning_rate = 5e-4
        config.learning_rate_end = 5e-6
        config.loss_type = 'mse'
        config.target_loss_index = None

        # build env
        env = gym.make(config.env_id)

        # build validation set
        # in this case just sequences of either 1s or 0s (const per sequence)
        # e.g.,
        # horizon = 4, and seeing 1s: [1 1 1 1] 
        # then after seeing [1 1], should predict a value from this point of 2
        # because that is the amount of reward expect to accrue in the future
        horizon = 4
        n_samples = 2
        n_timesteps = 2 # predict after seeing this many timesteps
        input_dim = 1
        # half ones and half neg ones
        x = np.ones((n_samples, n_timesteps, input_dim))
        x[int(n_samples / 2):] = -1
        # expect value to be how many timesteps have left * -1 or 1
        y = x[:,0,:] * (horizon - n_timesteps + 1)
        w = np.ones(n_samples)
        dataset = validation.Dataset(x, y, w)

        # run it
        summary_writer = tf.summary.FileWriter('/tmp/test')
        avg_loss = -1
        with tf.Session() as sess:
            trainer = async_td.AsyncTD(env, 0, config)
            sess.run(tf.global_variables_initializer())
            sess.run(trainer.sync)
            trainer.start(sess, summary_writer)
            global_step = sess.run(trainer.global_step)
            while global_step < config.n_global_steps:
                trainer.process(sess)
                if global_step % 10 == 0:
                    avg_loss = trainer.validate(sess, dataset)
                global_step = sess.run(trainer.global_step)

        self.assertTrue(avg_loss < .1)


         
class TestAsyncTDHeuristicDeterministicCase(unittest.TestCase):     

    def test_heuristic_deterministic_case(self):

        config = TestConfig()
        config.n_global_steps = 50000
        config.max_timesteps = 50
        config.env_id = 'BayesNetRiskEnv-v0'
        config.discount = 1. # 49. / 50
        config.value_dim = 5
        config.adam_beta1 = .9
        config.local_steps_per_update = 100
        config.hidden_layer_sizes = [128]
        config.learning_rate = 1e-3
        config.learning_rate_end = 5e-6
        config.loss_type = 'mse'
        config.target_loss_index = 1
        config.validation_dataset_filepath = '/Users/wulfebw/Dropbox/School/Stanford/research/risk/risk_prediction/data/experiments/heuristic_determinstic_1_lane_5_sec/data/subselect_proposal_prediction_data.h5'
        config.max_validation_samples = 1
        config.validate_every = 1000
        config.visualize_every = 10000
        config.summarize_features = True

        validation.transfer_dataset_settings_to_config(
            config.validation_dataset_filepath, config)

        config.base_bn_filepath = '/Users/wulfebw/Dropbox/School/Stanford/research/risk/risk_prediction/data/experiments/heuristic_determinstic_1_lane_5_sec/data/base_bn_filepath.h5'
        config.base_prop_filepath = '/Users/wulfebw/Dropbox/School/Stanford/research/risk/risk_prediction/data/experiments/heuristic_determinstic_1_lane_5_sec/data/prop_bn_filepath.h5'
        config.max_validation_samples = 1000

        # config.roadway_radius = 400.
        # config.roadway_length = 100.
        # config.lon_accel_std_dev = 0.
        # config.lat_accel_std_dev = 0.
        # config.overall_response_time = .2
        # config.lon_response_time = .0
        # config.err_p_a_to_i = .15
        # config.err_p_i_to_a = .3
        # config.max_num_vehicles = 50
        # config.min_num_vehicles = 50
        # config.hard_brake_threshold = -3.
        # config.hard_brake_n_past_frames = 2
        # config.min_base_speed = 30.
        # config.max_base_speed = 30.
        # config.min_vehicle_length = 5.
        # config.max_vehicle_length = 5.
        # config.min_vehicle_width = 2.5
        # config.max_vehicle_width = 2.5
        # config.min_init_dist = 10.
        # config.heuristic_behavior_type = "normal"

        # build env
        env = build_envs.create_env(config)
        dataset = validation.build_dataset(config, env)
        print('mean validation targets: {}'.format(np.mean(dataset.y, axis=0)))

        # run it
        summary_writer = tf.summary.FileWriter('/tmp/test')
        avg_loss = -1
        last_global_step_val = 0
        with tf.Session() as sess:
            trainer = async_td.AsyncTD(env, 0, config)
            sess.run(tf.global_variables_initializer())
            sess.run(trainer.sync)
            trainer.start(sess, summary_writer)
            global_step = sess.run(trainer.global_step)
            while global_step < config.n_global_steps:
                trainer.process(sess)
                if (global_step - last_global_step_val) > config.validate_every:
                    avg_loss = trainer.validate(sess, dataset)
                    last_global_step_val = global_step
                global_step = sess.run(trainer.global_step)
            

if __name__ == '__main__':
    unittest.main()
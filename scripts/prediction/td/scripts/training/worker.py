#!/usr/bin/env python
import argparse
import logging
import sys, signal
import time
import os
import tensorflow as tf

sys.path.append('../../')

import experiment_args
import prediction.async_td
import prediction.build_envs
import prediction.validation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def build_config(args):
    try:
        config_path = 'configs.{}'.format(args.config)
        config_module = __import__(config_path, fromlist=["configs"])
    except ImportError as e:
        print('error importing config file: {}'.format(args.config))
        print('make sure that the file exists')
        print('the argument should not end in \'.py\'')
        print('but should just be the name without \'.py\'')
        print('(though the actual file should of course have \'.py\')')
        raise(e)

    try:
        config = config_module.Config()
    except AttributeError as e:
        print('invalid config file: {}'.format(args.config))
        print('config file must have a class named \'Config\'')
        raise(e)

    # transfer settings from validation dataset if one exists
    if config.validation_dataset_filepath != '':
        config = prediction.validation.transfer_dataset_settings_to_config(
            config.validation_dataset_filepath, config)

    # if a key has been passed in as an arg, then that takes precedence 
    # over the values in the config; transfer them in here
    for (k,v) in args.__dict__.items():
        config.__dict__[k] = v

    # certain values have to be parsed manually
    # hidden layers sizes passed in as comma separated list
    if config.hidden_layer_sizes != '':
        dims = config.hidden_layer_sizes.split(',')
        dims = [int(dim) for dim in dims if dim != '']
        config.hidden_layer_sizes = dims
    # target_loss_index potentially None
    if config.target_loss_index == 'None':
        config.target_loss_index = None
    elif type(config.target_loss_index) == str:
        config.target_loss_index = int(config.target_loss_index)
    print(config.target_loss_index)
    print(type(config.target_loss_index))
    assert (config.target_loss_index is None or type(config.target_loss_index) == int)

    return config

def run(args, server):
    config = build_config(args)
    for k,v in config.__dict__.items():
        print(k)
        print(v)

    env = prediction.build_envs.create_env(config)
    dataset = prediction.validation.build_dataset(config, env)        
    trainer = prediction.async_td.AsyncTD(env, args.task, config)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables() 
        if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
        tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)
        logger.info("Finished initializing all parameters.")

    configproto = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = config.n_global_steps

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=configproto) as sess, sess.as_default():
        logger.info("running sync op")
        sess.run(trainer.sync)
        logger.info("starting trainer")
        trainer.start(sess, summary_writer)
        logger.info("gathering global step")
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        last_validation_global_step = 0
        while ( not sv.should_stop() 
                and (not num_global_steps 
                     or global_step < num_global_steps)):
            trainer.process(sess)
            if (global_step - last_validation_global_step > config.validate_every 
                    and dataset is not None):
                trainer.validate(sess, dataset)
                last_validation_global_step = global_step
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def main(_):
    """
    Setting up Tensorflow for data parallel work
    """
    parser = experiment_args.get_experiment_argparser('worker')
    args, unknown_args = parser.parse_known_args()
    print('unknown args: {}'.format(unknown_args))
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()

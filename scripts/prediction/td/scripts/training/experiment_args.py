import argparse

# args to use by default
# validation_dataset_filepath specifies a dataset, if that is provided then 
# the settings from it take precedence
# if not validation dataset is provided, then a config is loaded from config
# config filepath specifies defaults that 
def get_experiment_argparser(caller):
    parser = argparse.ArgumentParser(description=None)

    if caller == 'train':
        parser.add_argument('-w', '--num-workers', default=1, type=int,
                            help="Number of workers")
        parser.add_argument('-r', '--remotes', default=None,
                            help='The address of pre-existing VNC servers and '
                                 'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
        parser.add_argument('-e', '--env-id', type=str, default="RiskEnv-v0",
                            help="Environment id")
        parser.add_argument('-l', '--log-dir', type=str, default="/tmp/risk",
                            help="Log directory path")
        parser.add_argument('-n', '--dry-run', action='store_true',
                            help="Print out commands rather than executing them")
        parser.add_argument('-m', '--mode', type=str, default='tmux',
                            help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")
        parser.add_argument('-c', '--config', type=str, default='risk_env_config',
                            help="config filename, without \'.py\' extension.")

        # Add visualise tag
        parser.add_argument('--visualise', action='store_true',
                            help="Visualise the gym environment by running env.render() between each timestep")
    elif caller == 'worker':
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
        parser.add_argument('--task', default=0, type=int, help='Task index')
        parser.add_argument('--job-name', default="worker", help='worker or ps')
        parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
        parser.add_argument('--log-dir', default="/tmp/risk", help='Log directory path')
        parser.add_argument('--env-id', default="RiskEnv-v0", help='Environment id')
        parser.add_argument('-c', '--config', type=str, default='risk_env_config',
                        help="config filename, without \'.py\' extension. The default behavior is to match the config file to the choosen policy")
        parser.add_argument('-r', '--remotes', default=None,
                            help='References to environments to create (e.g. -r 20), '
                                 'or the address of pre-existing VNC servers and '
                                 'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

        # Add visualisation argument
        parser.add_argument('--visualise', action='store_true',
                            help="Visualise the gym environment by running env.render() between each timestep")

    # validation 
    parser.add_argument('--validation_dataset_filepath', default='', type=str,
                        help="filepath to validation dataset. Settings are loaded from this file")
    parser.add_argument('--validate_every', default=1000, type=int,
                        help="updates between validation steps")
    parser.add_argument('--max_validation_samples', default=500, type=int)

    # environment 
    parser.add_argument('--julia_env_id', type=str, default="BayesNetRiskEnv-v0",
                        help="Environment id")
    parser.add_argument('--max_timesteps', default=1000, type=int,
                        help="max timesteps in the environment before ending the episode")
    parser.add_argument('--prime_time', default=0., type=float,
                        help="Amount of time to prime the scene in the reset")
    parser.add_argument('--sampling_time', default=.1, type=float,
                        help="Amount of time to simulate per step (only .1 seconds is implemented currently)")

    # roadway
    parser.add_argument('--num_lanes', default=1, type=int,
                        help="number of lanes in roadway")
    parser.add_argument('--roadway_type', default="stadium", type=str,
                        help="type of roadway")
    parser.add_argument('--roadway_length', default=400., type=float,
                        help="length of roadway")
    parser.add_argument('--roadway_radius', default=100., type=float,
                        help="radius of roadway")

    # scene
    parser.add_argument('--max_num_vehicles', default=50, type=int,
                        help="max vehicles in scene")
    parser.add_argument('--min_num_vehicles', default=50, type=int,
                        help="min vehicles in scene")

    # generator
    parser.add_argument('--base_bn_filepath', default="", type=str,
                        help="filepath to base bayes net")
    parser.add_argument('--prop_bn_filepath', default="", type=str,
                        help="filepath to prop bayes net")

    # prediction
    parser.add_argument('--hidden_layer_sizes', default="64,64", type=str,
                        help="sizes of hidden layers, comma separated")
    parser.add_argument('--value_dim', default=5, type=int,
                        help="dimension of value function")
    parser.add_argument('--local_steps_per_update', default=100, type=int,
                        help="number of steps before running update, effective batch size")
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        help="initial learning rate")
    parser.add_argument('--learning_rate_end', default=5e-5, type=float,
                        help="final learning rate")
    parser.add_argument('--dropout_keep_prob', default=1., type=float,
                        help="prob drop units")
    parser.add_argument('--l2_reg', default=0., type=float,
                        help="l2 reg scale")
    parser.add_argument('--target_loss_index', default=None,
                        help="only compute loss against index is not None")
    parser.add_argument('--eps', default=1e-8, type=float,
                        help="")
    parser.add_argument('--loss_type', default='mse', type=str,
                        help="loss to use")
    parser.add_argument('--discount', default=1., type=float,
                        help="discount factor, should be (horizon-1)/horizon")
    parser.add_argument('--n_global_steps', default=100000000, type=int,
                        help="global steps to run experiment")
    parser.add_argument('--summary_every', default=11, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--adam_beta1', default=.99, type=float)
    parser.add_argument('--adam_beta2', default=.999, type=float)
    parser.add_argument('--rmsprop_decay', default=.9, type=float)
    parser.add_argument('--rmsprop_momentum', default=.9, type=float)

    # monitoring
    parser.add_argument('--viz_dir', default="videos/", type=str)
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--visualize_every', default=10000, type=int,
                        help="# episodes between rendering")
    parser.add_argument('--summarize_features', default=True, action='store_true',
                        help="add feature summaries to tensorboard")
    return parser

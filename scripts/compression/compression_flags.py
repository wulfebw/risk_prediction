
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# training constants
tf.app.flags.DEFINE_integer('batch_size', 
                            32,
                            """Number of samples in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 
                            100,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_string('snapshot_dir', 
                           '../../data/snapshots/test/',
                           """Path to directory where to save weights.""")
tf.app.flags.DEFINE_string('viz_dir', 
                           '../../data/visualizations/test/',
                           """Path to directory where to save visualizations.""")
tf.app.flags.DEFINE_string('summary_dir', 
                           '../../data/summaries/test',
                           """Path to directory where to save summaries.""")
tf.app.flags.DEFINE_string('julia_weights_filepath', 
                           '../../data/networks/test.weights',
                           """Path to file where to save julia weights.""")
tf.app.flags.DEFINE_integer('save_every', 
                            1000000,
                            """Number of epochs between network saves.""")
tf.app.flags.DEFINE_bool('verbose', 
                            True,
                            """Wether or not to print out progress.""")
tf.app.flags.DEFINE_integer('debug_size', 
                            None,
                            """Debug size to use.""")
tf.app.flags.DEFINE_integer('random_seed', 
                            1,
                            """Random seed value to use.""")
tf.app.flags.DEFINE_bool('load_network', 
                            False,
                            """Wether or not to load from a saved network.""")
tf.app.flags.DEFINE_integer('log_summaries_every', 
                            2,
                            """Number of batches between logging summaries.""")
tf.app.flags.DEFINE_integer('save_weights_every', 
                            1,
                            """Number of batches between logging summaries.""")
tf.app.flags.DEFINE_bool('balanced_class_loss', 
                            False,
                            """Whether or not to balance the classes in 
                            classification loss by reweighting.""")
tf.app.flags.DEFINE_integer('target_index', 
                            None,
                            """Target index to fit exclusively if set (zero-based).
                            This must be accompanied by setting output_dim to 1.""")

# network constants
tf.app.flags.DEFINE_integer('max_norm', 
                            100000,
                            """Maximum gradient norm.""")
tf.app.flags.DEFINE_integer('hidden_dim', 
                            64,
                            """Hidden units in each hidden layer.""")
tf.app.flags.DEFINE_integer('num_hidden_layers', 
                            2,
                            """Number of hidden layers.""")
tf.app.flags.DEFINE_string('hidden_layer_dims', 
                            '',
                            """Hidden layer sizes, empty list means use hidden_dim.""")
tf.app.flags.DEFINE_float('learning_rate', 
                            0.0005,
                            """Initial learning rate to use.""")
tf.app.flags.DEFINE_float('decrease_lr_threshold', 
                            .001,
                            """Percent decrease in validation loss below 
                            which the learning rate will be decayed.""")
tf.app.flags.DEFINE_float('decay_lr_ratio', 
                            1.,
                            """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('min_lr', 
                            .000005,
                            """Minimum learning rate value.""")
tf.app.flags.DEFINE_string('loss_type', 
                           'ce',
                           """Type of loss to use {mse, ce}.""")
tf.app.flags.DEFINE_string('task_type', 
                           'regression',
                           """Type of task {regression, classification}.""")
tf.app.flags.DEFINE_integer('num_target_bins', 
                            None,
                            """Number of bins into which to discretize targets.""")
tf.app.flags.DEFINE_float('dropout_keep_prob', 
                            1.,
                            """Probability to keep a unit in dropout.""")
tf.app.flags.DEFINE_boolean('use_batch_norm', 
                            False,
                            """Whether to use batch norm (True removes dropout).""")
tf.app.flags.DEFINE_float('l2_reg', 
                            0.0,
                            """Probability to keep a unit in dropout.""")
tf.app.flags.DEFINE_float('eps', 
                            1e-8,
                            """Minimum probability value.""")

# dataset constants
tf.app.flags.DEFINE_string('dataset_filepath',
                            '../../data/datasets/risk.jld',
                            'Filepath of dataset.')
tf.app.flags.DEFINE_integer('input_dim', 
                            276,
                            """Dimension of input.""")
tf.app.flags.DEFINE_integer('timesteps', 
                            1,
                            """Number of input timesteps.""")
tf.app.flags.DEFINE_integer('output_dim', 
                            5,
                            """Dimension of output.""")
tf.app.flags.DEFINE_bool('use_priority', 
                            False,
                            """Wether or not to use a prioritized dataset.""")
tf.app.flags.DEFINE_float('priority_alpha', 
                            0.25,
                            """Alpha parameter for prioritization.""")
tf.app.flags.DEFINE_float('priority_beta', 
                            1.0,
                            """Beta parameter for prioritization.""")

# bootstrapping constants
tf.app.flags.DEFINE_integer('bootstrap_iterations', 
                            10,
                            """Number of iterations of collecting a bootstrapped dataset and fitting it.""")
tf.app.flags.DEFINE_integer('num_proc', 
                            1,
                            """Number of processes to use for dataset collection.""")
tf.app.flags.DEFINE_integer('num_scenarios', 
                            1,
                            """Number of scenarios in each dataset.""")
tf.app.flags.DEFINE_string('initial_network_filepath',
                            'none',
                            'Filepath of initial network or none.')
tf.app.flags.DEFINE_string('run_filepath',
                            'run_collect_debug_dataset.jl',
                            'Filepath to run file.')


def custom_parse_flags(flags):
    # hidden layer dims
    if flags.hidden_layer_dims != '':
        dims = flags.hidden_layer_dims.split(' ')
        dims = [int(dim) for dim in dims]
    else:
        dims = [flags.hidden_dim for _ in range(flags.num_hidden_layers)]

    flags.hidden_layer_dims = dims
    print('Building network with hidden dimensions: {}'.format(
            flags.hidden_layer_dims))

    # task
    if flags.num_target_bins is not None:
        flags.task_type = 'classification'
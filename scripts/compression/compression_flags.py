import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

flags_list = [
# training constants
    ('batch_size', 
        32, 
        """Number of samples in a batch."""),
    ('num_epochs', 
        100, 
        """Number of training epochs."""),
    ('snapshot_dir', 
        '../../data/snapshots/test/',
        """Path to directory where to save weights."""),
    ('viz_dir_parent',
        '../../data/visualizations/',
        """Path to directory where to save visualizations."""),
    ('viz_dir',
        '../../data/visualizations/test/',
        """Path to directory where to save visualizations."""),
    ('summary_dir',
        '../../data/summaries/test',
        """Path to directory where to save summaries."""),
    ('julia_weights_filepath',
        '../../data/networks/test.weights',
        """Path to file where to save julia weights."""),
    ('save_every',
        1000000,
        """Number of epochs between network saves."""),
    ('verbose', 
        True, 
        """Wether or not to print out progress."""),
    ('debug_size', 
        None, 
        """Debug size to use."""),
    ('random_seed', 
        1, 
        """Random seed value to use."""),
    ('load_network', 
        False, 
        """Wether or not to load from a saved network."""),
    ('log_summaries_every', 
        2, 
        """Number of batches between logging summaries."""),
    ('save_weights_every',
        1,
        """Number of batches between logging summaries."""),
    ('balanced_class_loss',
        False,
        """Whether or not to balance the classes in classification loss by reweighting."""),
    ('target_index',
        None,
        """Target index to fit exclusively if set (zero-based). 
        This must be accompanied by setting output_dim to 1."""),
    ('shuffle_data',
        True,
        """Whether to shuffle the data in loading it."""),

# network constants
    ('max_norm', 
        100000, 
        """Maximum gradient norm."""),
    ('hidden_dim', 
        64, 
        """Hidden units in each hidden layer."""),
    ('num_hidden_layers', 
        2, 
        """Number of hidden layers."""),
    ('hidden_layer_dims', 
        '', 
        """Hidden layer sizes, empty list means use hidden_dim."""),
    ('learning_rate',
        0.001,
         """Initial learning rate to use."""),
    ('decrease_lr_threshold',
        .001,
        """Percent decrease in validation loss below which the learning rate will be decayed."""),
    ('decay_lr_ratio',
        1.,
         """Learning rate decay factor."""),
    ('min_lr', 
        .000005, 
        """Minimum learning rate value."""),
    ('loss_type',
        'ce',
        """Type of loss to use {mse, ce}."""),
    ('task_type',
        'regression',
        """Type of task {regression, classification}."""),
    ('num_target_bins',
        None,
        """Number of bins into which to discretize targets."""),
    ('dropout_keep_prob',
        1.,
         """Probability to keep a unit in dropout."""),
    ('use_batch_norm',
        False,
         """Whether to use batch norm (True removes dropout)."""),
    ('l2_reg',
        0.0,
        """Probability to keep a unit in dropout."""),
    ('eps',
        1e-8,
        """Minimum probability value."""),

# dataset constants
    ('dataset_filepath',
        '../../data/datasets/risk.jld',
        'Filepath of dataset.'),
    ('input_dim',
        276,
        """Dimension of input."""),
    ('timesteps',
        1,
        """Number of input timesteps."""),
    ('output_dim',
        5,
        """Dimension of output."""),
    ('use_priority',
        False,
        """Wether or not to use a prioritized dataset."""),
    ('priority_alpha',
        0.25,
        """Alpha parameter for prioritization."""),
    ('priority_beta',
        1.0,
         """Beta parameter for prioritization."""),
    ('use_likelihood_weights',
        False,
        """Wether or not to load likelihood ratio weights."""),

# bootstrapping constants
    ('bootstrap_iterations',
        10,
        """Number of iterations of collecting a bootstrapped dataset and fitting it."""),
    ('num_proc',
        1,
        """Number of processes to use for dataset collection."""),
    ('num_scenarios',
        1,
         """Number of scenarios in each dataset."""),
    ('initial_network_filepath',
        'none',
        """Filepath of initial network or none."""),
    ('run_filepath',
        'run_collect_debug_dataset.jl',
        """Filepath to run file.""" ),
]

for  flag_name, default_value, docstring in flags_list:
    if type(default_value) == int:
        tf.app.flags.DEFINE_integer(flag_name, default_value, docstring)
    elif type(default_value) == str:
        tf.app.flags.DEFINE_string(flag_name, default_value, docstring)
    elif type(default_value) == bool:
        tf.app.flags.DEFINE_bool(flag_name, default_value, docstring)
    elif type(default_value) == float:
        tf.app.flags.DEFINE_float(flag_name, default_value, docstring)

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

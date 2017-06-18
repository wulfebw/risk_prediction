"""
Functions for defining variables in a network.
"""

import tensorflow as tf

def get_weight_initializer(activation):
    """
    Description:
        - Given an activation function, return a weight
            initializer that works well for that activation
            function.

            Relu: "Delving Deep into Rectifiers:
            Surpassing Human-Level Performance on ImageNet
            Classification" 
            Source: https://arxiv.org/abs/1502.01852

            Tanh: 

    Args:
        - activation: string indicating the activation function.
            one of {'relu', 'tanh'}

    Returns:
        - initializer: a tensorflow weight initializer.
    """
    if activation == 'relu':
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
    elif activation == 'tanh':
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode='FAN_AVG', uniform=False)
    else:
        raise ValueError('invalid activation: {}'.format(activation))
    return initializer

def get_bias_initializer(activation):
    """
    Description: 
        - Given an activation function, return a bias
            initializer that works well for that activation
            function.

    Args:
        - activation: string denoting activation, 
            one of {'relu', 'tanh'}

    Returns:
        - initializer: tensorflow bias initializer
    """
    if activation == 'relu':
        initializer = tf.constant_initializer(0.1)
    elif activation == 'tanh':
        initializer = tf.constant_initializer(0.0)
    else:
        raise ValueError('invalid activation: {}'.format(activation))
    return initializer


def build_feed_forward_network(input_ph, dropout_ph, flags):
    """
    Description:
        - Builds a feed forward network with relu units.

    Args:
        - input_ph: placeholder for the inputs
            shape = (batch_size, input_dim)
        - dropout_ph: placeholder for dropout value
        - flags: config values

    Returns:
        - scores: the scores for the target values
    """

    # build initializers specific to relu
    weights_initializer = get_weight_initializer('relu')
    bias_initializer = get_bias_initializer('relu')

    # build regularizers
    weights_regularizer = tf.contrib.layers.l2_regularizer(flags.l2_reg)

    # build hidden layers
    # if layer dims not set individually, then assume all the same dim
    hidden_layer_dims = flags.hidden_layer_dims
    if len(hidden_layer_dims) == 0:
        hidden_layer_dims = [flags.hidden_dim 
            for _ in range(flags.num_hidden_layers)]

    hidden = input_ph
    for (lidx, hidden_dim) in enumerate(hidden_layer_dims):
        hidden = tf.contrib.layers.fully_connected(hidden, 
            hidden_dim, 
            activation_fn=tf.nn.relu,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            biases_initializer=bias_initializer)
        # tf.histogram_summary("layer_{}_activation".format(lidx), hidden)
        if flags.use_batch_norm:
            hidden = tf.contrib.layers.batch_norm(hidden)
        hidden = tf.nn.dropout(hidden, dropout_ph)

    # build output layer
    output_dim = flags.output_dim
    if flags.task_type == 'classification':
        output_dim *= flags.num_target_bins
    scores = tf.contrib.layers.fully_connected(hidden, 
            output_dim, 
            activation_fn=None,
            weights_regularizer=weights_regularizer)

    return scores

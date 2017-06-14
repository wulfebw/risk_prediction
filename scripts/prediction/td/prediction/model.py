import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], 
        initializer=initializer)
    b = tf.get_variable(name + "/b", [size], 
        initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

class LSTMPredictor(object):
    """
    Wrapper around an LSTM with an obs space mapping.
    Notes:
        - must pass in state_in
        - initially should pass state_init
        - vf (value function) is then a list of outputs, one for each timestep
        - to extract the vf prediction after seeing some number of 
            observations, you should select vf[-1] because the value function 
            is expressed with the first dimension corresponding to time
    """
    def __init__(self, ob_space, config):
        self.config = config
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), 'x')
        self.dropout_keep_prob_ph = tf.placeholder(tf.float32, 
            shape=(), 
            name='dropout_keep_prob_ph'
        )

        # hidden layers before lstm
        for l, size in enumerate(config.hidden_layer_sizes):
            x = tf.nn.elu(linear(x, size, "{}_".format(l), 
                normalized_columns_initializer(0.01)))
            x = tf.nn.dropout(x, self.dropout_keep_prob_ph)
        size = config.hidden_layer_sizes[-1]
        
        # introduce a "fake" batch dimension of 1 to LSTM over time dim
        x = tf.expand_dims(x, [0])
        step_size = tf.shape(self.x)[:1]
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]
        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        x = tf.reshape(lstm_outputs, [-1, size])
        self.vf = linear(x, config.value_dim, "value", normalized_columns_initializer(1.0))
        self.var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def features(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.state_out, {
            self.x: [ob], 
            self.state_in[0]: c, 
            self.state_in[1]: h,
            self.dropout_keep_prob_ph: 1.
        })

    def value(self, ob, c, h, sequence=False):
        if not sequence:
            ob = [ob]
        sess = tf.get_default_session()
        v = sess.run(self.vf, {
            self.x: ob, 
            self.state_in[0]: c, 
            self.state_in[1]: h,
            self.dropout_keep_prob_ph: 1.
        })
        # when not a sequence input, there's only a single value and we 
        # extract it from the output list
        if not sequence:
            v = v[0]
        # when computing the value of a sequence we take the last output
        else:
            v = v[-1]

        # convert to probability form if necessary
        if self.config.loss_type == 'log_mse':
            v = np.exp(v)
        elif self.config.loss_type == 'ce':
            v = 1 / (1 + np.exp(-v))
        return v
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
    def __init__(self, ob_space, config):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        # hidden layers before lstm
        for l, size in enumerate(config.hidden_layer_sizes):
            x = tf.nn.elu(linear(x, size, "{}_".format(l), 
                normalized_columns_initializer(0.01)))
            x = tf.nn.dropout(x, config.dropout_keep_prob)
            x = tf.contrib.layers.batch_norm(x)
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
        x = tf.nn.elu(linear(x, config.value_dim, "hidden_value", normalized_columns_initializer(1.0)))
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
            self.state_in[1]: h
        })

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {
            self.x: [ob], 
            self.state_in[0]: c, 
            self.state_in[1]: h
        })[0]



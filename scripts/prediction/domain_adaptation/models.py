
import tensorflow as tf

def encoder(
        inputs, 
        scope, 
        reuse=False,
        hidden_layer_dims=[64,64],
        dropout_keep_prob=1.):
    '''returns encoding'''
    with tf.variable_scope(scope, reuse=reuse):
        hidden = inputs
        for dim in hidden_layer_dims:
            hidden = tf.contrib.layers.fully_connected(hidden, 
                dim, 
                activation_fn=tf.nn.relu)
            hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        return hidden 
    
def classifier(
        inputs, 
        output_dim, 
        scope, 
        reuse=False):
    '''returns class scores'''
    with tf.variable_scope(scope, reuse=reuse):
        return tf.contrib.layers.fully_connected(inputs, output_dim, activation_fn=None)
    
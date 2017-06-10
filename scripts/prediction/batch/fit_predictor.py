
import copy
import numpy as np
np.set_printoptions(suppress=True, precision=8)
import os
import sys
import tensorflow as tf

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

from . import prediction_metrics
from . import prediction_flags
from . import dataset
from . import dataset_loaders
import neural_networks.feed_forward_neural_network as ffnn
import neural_networks.recurrent_neural_network as rnn
import neural_networks.utils

FLAGS = prediction_flags.FLAGS

def main(argv=None):
    # custom parse of flags for list input
    prediction_flags.custom_parse_flags(FLAGS)

    # set random seeds
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    # load dataset
    input_filepath = FLAGS.dataset_filepath
    data = dataset_loaders.risk_dataset_loader(
        input_filepath, shuffle=FLAGS.shuffle_data, train_split=.8, 
        debug_size=FLAGS.debug_size, timesteps=FLAGS.timesteps,
        num_target_bins=FLAGS.num_target_bins, 
        balanced_class_loss=FLAGS.balanced_class_loss, 
        target_index=FLAGS.target_index,
        load_likelihood_weights=FLAGS.use_likelihood_weights)

    if FLAGS.use_priority:
        d = priority_dataset.PrioritizedDataset(data, FLAGS)
    else:
        if FLAGS.balanced_class_loss or FLAGS.use_likelihood_weights:
            d = dataset.WeightedDataset(data, FLAGS)
        else:
            d = dataset.Dataset(data, FLAGS)

    print('training set size: {}'.format(len(data['x_train'])))
    print('means:\n{}\n{}'.format(
        np.mean(d.data['y_train'], axis=0),
        np.mean(d.data['y_val'], axis=0)))
    y = copy.deepcopy(d.data['y_val'])
    y[y==0.] = 1e-8
    y[y==1.] = 1 - 1e-8
    prediction_metrics.regression_score(y, np.mean(y, axis=0), 'baseline')
    prediction_metrics.regression_score(y, y, 'correct')

    # fit the model
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
        # if the timestep dimension is > 1, use recurrent network
        if FLAGS.timesteps > 1:
            if FLAGS.task_type == 'classification':
                network = rnn.ClassificationRecurrentNeuralNetwork(session, FLAGS)
            else:
                network = rnn.RecurrentNeuralNetwork(session, FLAGS)
        else:
            if FLAGS.task_type == 'classification':
                if FLAGS.balanced_class_loss or FLAGS.use_likelihood_weights:
                    network = ffnn.WeightedClassificationFeedForwardNeuralNetwork(session, FLAGS)
                else:
                    network = ffnn.ClassificationFeedForwardNeuralNetwork(session, FLAGS)
            else:
                network = ffnn.FeedForwardNeuralNetwork(session, FLAGS)
        network.fit(d)

        # save weights to a julia-compatible weight file
        neural_networks.utils.save_trainable_variables(
            FLAGS.julia_weights_filepath, session, data)

        # evaluate the fit
        prediction_metrics.evaluate_fit(network, data, FLAGS)

if __name__ == '__main__':
    tf.app.run()

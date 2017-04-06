import copy
import numpy as np
np.set_printoptions(suppress=True, precision=8)
import os
import sys
import tensorflow as tf

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

from compression import compression_metrics
from compression import compression_flags
import dataset_loaders
import neural_networks.feed_forward_neural_network as ffnn
import neural_networks.utils

FLAGS = compression_flags.FLAGS

def main(argv=None):
    compression_flags.custom_parse_flags(FLAGS)
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    data = dataset_loaders.risk_dataset_loader(
        FLAGS.dataset_filepath, shuffle=True, train_split=.9, 
        debug_size=FLAGS.debug_size, timesteps=FLAGS.timesteps,
        num_target_bins=FLAGS.num_target_bins, balanced_class_loss=FLAGS.balanced_class_loss, target_index=FLAGS.target_index)
    x = data['x_train']
    y = data['y_train']
    eps = 1e-8
    y[y==0.] = eps
    y[y==1.] = 1 - eps

    base_dir = '/home/sisl/blake/risk_prediction/data/snapshots'
    snapshot_dir_names = ['mc_{}'.format(c) for c in [1,2,4,8,16,32]]
    snapshot_dirs = [os.path.join(base_dir, name) for name in snapshot_dir_names]
    print(snapshot_dirs)
    r2s = []
    with tf.Session() as session:
        network = ffnn.FeedForwardNeuralNetwork(session, FLAGS)
        for snapshot_dir in snapshot_dirs:
            FLAGS.snapshot_dir = snapshot_dir
            network.load()
            y_pred = network.predict(x)
            y_pred[y_pred < eps] = eps
            y_pred[y_pred > 1 - eps] = 1 - eps
            ll = np.sum(y * np.log(y_pred)) + np.sum((1 - y) * np.log(1 - y_pred)) 
            y_null = np.mean(y, axis=0, keepdims=True)
            y_null[y_null < eps] = eps
            y_null[y_null > 1 - eps] = 1 - eps
            ll_null = np.sum(y * np.log(y_null)) + -np.sum((1 - y) * np.log(1 - y_null))
            mcfadden_r2 = 1 - ll / ll_null
            mse = np.sum((y_pred - y) ** 2)
            mean = np.mean(y_pred)
            r2s.append((mcfadden_r2, mse, mean))

    for (r2, snapshot_dir) in zip(r2s, snapshot_dirs):
        print(snapshot_dir)
        print(r2)
        print()
            

if __name__ == '__main__':
    tf.app.run()

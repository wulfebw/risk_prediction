
import argparse
import numpy as np
np.set_printoptions(precision=6, suppress=True)
import os
from sklearn import dummy
from sklearn import ensemble
from sklearn import multioutput
import sys
import time

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset_loaders
from compression import compression_metrics

MODEL_TYPES = [
    'gradient_boosting',
    'dummy_stratified',
    'dummy_most_frequent', 
    # 'random_forest'
]

def fit(model, data, viz_dir, name):
    print("fitting {}".format(model))
    x_train, y_train = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_probs = model.predict_proba(x_val).reshape(len(y_pred), 5, 2)
    compression_metrics.classification_score(y_val, y_pred, y_probs, 
        np.ones(len(y_pred)), name, viz_dir)

def build_model(model_type, num_targets = 1):
    if model_type == 'gradient_boosting':
        base = ensemble.GradientBoostingClassifier(n_estimators=100, verbose=True)
    elif model_type == 'random_forest':
        base = ensemble.RandomForestClassifier()
    elif model_type == 'dummy_stratified':
        base = dummy.DummyClassifier('stratified')
    elif model_type == 'dummy_most_frequent':
        base = dummy.DummyClassifier('most_frequent')
    else:
        raise(ValueError('invalid model type: {}'.format(model_type)))

    # multiple outputs in the dataset => fit a separate regressor to each
    if num_targets > 1:
        return multioutput.MultiOutputClassifier(base)
    else:
        return base

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_type', 
        default='all')
    parser.add_argument('-f', dest='dataset_filepath', 
        default='../../data/datasets/may/ngsim_5_sec.h5')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # in case things get a bit crazy
    np.random.seed(1)

    # parse inputs
    opts = parse_args()

    # load the dataset
    data = dataset_loaders.risk_dataset_loader(
        opts.dataset_filepath, shuffle=True, train_split=.9, 
        debug_size=None, timesteps=1, num_target_bins=2)

    # build the model
    if len(data['y_train'].shape) > 1:
        _, num_targets = data['y_train'].shape
    else:
        num_targets = 1
    if opts.model_type == 'all':
        models = [build_model(mt, num_targets) for mt in MODEL_TYPES]
        names = [mt for mt in MODEL_TYPES]
    else:
        model = build_model(opts.model_type, num_targets)
        name = opts.model_type

    # fit the model
    viz_dir = '../../data/visualizations/baseline/'
    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)
    st = time.time()
    if opts.model_type == 'all':
        results = [fit(m, data, viz_dir, n) for (m,n) in zip(models, names)]
    else:
        results = fit(model, data, viz_dir, name)
    et = time.time()
    print('model fitting took {} seconds'.format(et - st))

    # display results
    if opts.model_type == 'all':
        results = sorted(zip(results, MODEL_TYPES), reverse=True)
        for (r, l) in results:
            print('{}: {}'.format(l,r))
    else:
        print(results)

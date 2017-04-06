
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import os
import sys
import sklearn.metrics

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset_loaders

def report_poorly_performing_indices_features(idxs, data):
    batch_idxs = data['batch_idxs']
    seeds = data['seeds']
    for idx in idxs:
        for i, b in enumerate(batch_idxs):
            if b > idx:
                break
        seed = seeds[i]
        if i > 0:
            veh_idx = idx - batch_idxs[i - 1] + 1
        else:
            veh_idx = idx + 1
        print('seed: {}\tveh idx: {}'.format(seed, veh_idx))
        print('targets: {}'.format(data['y_train'][idx]))
        print('seed num veh: {}'.format(batch_idxs[i] - batch_idxs[i-1]))

def classification_score(y, y_pred, probs, name, flags):
    print('\nclassification results for {}'.format(name))
    for tidx in range(y.shape[1]):
        print('\n###### target: {}'.format(tidx))
        print(sklearn.metrics.classification_report(y[:,tidx], y_pred[:,tidx]))
        print(sklearn.metrics.confusion_matrix(y[:,tidx], y_pred[:,tidx]))
        if probs.shape[-1] == 2:
            pos_idx = 1 # means 1 is the positive class

            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y[:,tidx], probs[:,tidx,pos_idx], pos_label=pos_idx)
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label='{}_{} (area = {})'.format(name, tidx, roc_auc))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('roc for {} target {}'.format(name, tidx))
            output_filepath = os.path.join(
                flags.viz_dir, 'roc_{}_{}.png'.format(name, tidx))
            plt.legend()
            plt.savefig(output_filepath)
            plt.clf()

            precision, recall, _ = sklearn.metrics.precision_recall_curve(y[:,tidx], probs[:,tidx,pos_idx])
            avg_precision = sklearn.metrics.average_precision_score(y[:,tidx], probs[:,tidx,pos_idx])
            plt.plot(recall, precision, label='{}_{} (area = {})'.format(name, tidx, avg_precision))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('prc for {} target {}'.format(name, tidx))
            output_filepath = os.path.join(
                flags.viz_dir, 'prc_{}_{}.png'.format(name, tidx))
            plt.legend()
            plt.savefig(output_filepath)
            plt.clf()
            
        print('######\n')

def regression_score(y, y_pred, name, data=None, eps=1e-16, 
        y_null=None):
    # prevent overflow during the sum of the log terms
    y_pred = y_pred.astype(np.float128)
    # also threshold values to prevent log exception (throws off loss value)
    y_pred[y_pred < eps] = eps
    y_pred[y_pred > 1 - eps] = 1 - eps
    
    np.sum(y * np.log(y_pred))
    np.sum((1 - y) * np.log(1 - y_pred)) 
    ll = np.sum(y * np.log(y_pred)) + np.sum((1 - y) * np.log(1 - y_pred)) 
    ce = -ll
    mse = np.sum((y - y_pred) ** 2)
    r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum()

    # worst indices
    if len(np.shape(y)) > 1:
        max_mse_idx = np.argmax(np.sum((y - y_pred) ** 2, axis=1))
        max_ce_idx = np.argmax(np.sum(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)), axis=1))
    else:
        max_mse_idx = np.argmax((y - y_pred) ** 2)
        max_ce_idx = np.argmax(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    num_samples = len(y_pred)
    print("\n{} final cross entropy: {}".format(name, ce / num_samples))
    print("{} final mse: {}".format(name, mse / num_samples))
    print("{} final r2: {}".format(name, r2))
    if len(y_pred) > max_ce_idx and len(y_pred) > max_mse_idx:
        print("{} worst ce (idx {}):\n y: {} y_pred: {}".format(
            name, max_ce_idx, y[max_ce_idx], y_pred[max_ce_idx]))
        print("{} worst mse (idx {}):\n y: {} y_pred: {}".format(
            name, max_mse_idx, y[max_mse_idx], y_pred[max_mse_idx]))

    # convert to julia format the worst indices
    if data is not None:
        num_report = 5
        print('\noverall poorly predicted')
        idxs = np.argsort(np.sum(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)), axis=1))[-num_report:]
        report_poorly_performing_indices_features(idxs, data)
        print('\nrear end collisions poorly predicted')
        idxs = np.argsort(np.sum(-(y[:,1:3] * np.log(y_pred[:,1:3]) + (1 - y[:,1:3]) * np.log(1 - y_pred[:,1:3])), axis=1))[-num_report:]
        report_poorly_performing_indices_features(idxs, data)
        print('\nhard brakes poorly predicted')
        idxs = np.argsort(-(y[:,3] * np.log(y_pred[:,3]) + (1 - y[:,3]) * np.log(1 - y_pred[:,3])))[-num_report:]
        report_poorly_performing_indices_features(idxs, data)

    # psuedo r^2 and other metrics
    if y_null is not None:
        if len(np.shape(y_null)) > 0:
            y_null[y_null < eps] = eps
            y_null[y_null > 1 - eps] = 1 - eps

        null_ll = np.sum(y * np.log(y_null)) + -np.sum((1 - y) * np.log(1 - y_null))
        mcfadden_r2 = 1 - ll / null_ll
        tjur_r2 = np.mean(y_pred[y>=.5], axis=0) - np.mean(y_pred[y<.5], axis=0)
        y_class = np.zeros(y.shape)
        y_class[y>.5] = 1
        y_class = y_class.flatten()
        y_pred_class = (copy.deepcopy(y_pred) + .5).astype(int)
        y_pred_class = y_pred_class.flatten()
        acc = len(np.where(y_class == y_pred_class)[0]) / np.prod(y_class.shape)
        prec_idxs = np.where(y_pred_class == 1)[0]
        prec = recall = len(np.where(y_class[prec_idxs] == 1)[0])
        if len(prec_idxs) > 0:
            prec /= len(prec_idxs)
            normalization_count = len(np.where(y_class == 1)[0])
            if normalization_count > 0:
                recall /= normalization_count
            else:
                recall = 0
        else:
            prec = 0
            recall = 0

        print("mcfadden r^2: {}\tll: {}\tnull ll: {}".format(mcfadden_r2, ll, null_ll))
        print("tjur_r2: {}".format(tjur_r2))
        print("acc: {}\tprecision: {}\trecall: {}".format(acc, prec, recall))
        
    return ce, mse, r2

def evaluate_classification_fit(network, data, flags):
    # train
    y_pred, y_probs = network.predict(data['x_train'])
    y = data['y_train']
    y_null = np.mean(y, axis=0)
    classification_score(y, y_pred, y_probs, 'train', flags)

    # validation
    y_pred, y_probs = network.predict(data['x_val'])
    y = data['y_val']
    y_null = np.mean(y, axis=0)
    classification_score(y, y_pred, y_probs, 'val', flags)

def evaluate_regression_fit(network, data, flags):
    # final train loss
    y_pred = network.predict(data['x_train'])
    y = data['y_train']
    y_null = np.mean(y, axis=0)
    regression_score(y, y_pred, 'train', y_null=y_null)

    # final validation loss
    y_pred = network.predict(data['x_val'])
    y = data['y_val']
    y_null = np.mean(y, axis=0)
    regression_score(y, y_pred, 'val', y_null=y_null)

    y_pred = network.predict(data['x_val'])
    y_pred = y_pred[:, 3]
    y = data['y_val'][:, 3]
    y_null = np.mean(y, axis=0)
    regression_score(y, y_pred, 'hard brake', y_null=y_null)

    data = dataset_loaders.risk_dataset_loader(
        flags.dataset_filepath, shuffle=False, train_split=1., 
        debug_size=flags.debug_size, timesteps=flags.timesteps)
    y_pred = network.predict(data['x_train'])
    regression_score(data['y_train'], y_pred, 'unshuffled', data)

def compare_classification_output(network, data, flags, num_samples=10):
    y_idxs = np.where(np.sum(data['y_val'][:10000], axis=1) > 1e-4)[0]
    y_idxs = np.random.permutation(y_idxs)[:num_samples]
    y_pred, pred_probs = network.predict(data['x_val'][y_idxs])
    for y_pred_s, y_s in zip(y_pred, data['y_val'][y_idxs]):
        print('actual:{}\npredicted:{}'.format(y_s, y_pred_s))

def evaluate_fit(network, data, flags):
    if not os.path.exists(flags.viz_dir):
        os.mkdir(flags.viz_dir)
    
    if flags.task_type == 'classification':
        compare_classification_output(network, data, flags)
        evaluate_classification_fit(network, data, flags)
    else:
        evaluate_regression_fit(network, data, flags)

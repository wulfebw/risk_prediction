
import numpy as np
import os

import matplotlib
backend = 'Agg' if os.system == 'linux' else 'TkAgg'
matplotlib.use(backend)
import matplotlib.pyplot as plt

def hist_targets(
        filepath, 
        src_targets_train, 
        src_targets_val,
        tgt_targets_train, 
        tgt_targets_val,
        label, 
        bins=20):
    plt.figure(figsize=(8,8))
    plt.hist([
        src_targets_train[:,1],
        src_targets_val[:,1],
        tgt_targets_train[:,1],
        tgt_targets_val[:,1]
    ], bins=bins, normed=True, label=[
        'src_targets_train',
        'src_targets_val',
        'tgt_targets_train',
        'tgt_targets_val'
    ])
    plt.title('{} histogram'.format(label))
    plt.legend()
    plt.savefig(filepath)
    plt.clf()

def hist_features(
        filepath_template, 
        src_features_train, 
        src_features_val,
        tgt_features_train, 
        tgt_features_val,
        labels, 
        bins=20):
    n = len(labels)
    for i in range(n):
        plt.figure(figsize=(8,4))
        plt.hist([
            src_features_train[:,i], 
            src_features_val[:,i],
            tgt_features_train[:,i],
            tgt_features_val[:,i]
        ], bins=bins, normed=True, label=[
            'src_features_train',
            'src_features_val',
            'tgt_features_train',
            'tgt_features_val'
        ])
        plt.title('{} histogram'.format(labels[i]))
        plt.legend()
        plt.savefig(filepath_template.format(labels[i]))
        plt.clf()

def visualize(data, vis_dir):
    hist_targets(
        os.path.join(vis_dir, '{}.png'.format(data['target_names'])),
        data['src_y_train'],
        data['src_y_val'],
        data['tgt_y_train'],
        data['tgt_y_val'],
        label=data['target_names']
    )
    # hist_features(
    #     os.path.join(vis_dir, '{}.png'),
    #     data['src_x_train'],
    #     data['src_x_val'],
    #     data['tgt_x_train'],
    #     data['tgt_x_val'],
    #     labels=data['feature_names']
    # )

def load_results(filepath):
    return np.load(filepath).item()

def extract_tgt_results(res):
    keys = list(sorted(res.keys()))
    sizes = []
    losses = []
    for k in keys:
        sizes.append(k)
        losses.append(res[k]['val_info']['tgt_info']['task_loss'])
    return sizes, losses

def visualize_results(res, label='', c='red'):
    sizes, losses = extract_tgt_results(res)
    plt.plot(sizes, losses, label=label, c=c)
    ax = plt.gca()
    ax.set_ylim([-.1,2])

if __name__ == '__main__':
    filepath = '../../../data/datasets/da_results_with_adapt_40980.npy'
    res = load_results(filepath)
    visualize_results(res, label='with adaptation', c='blue')

    filepath = '../../../data/datasets/da_results_without_adapt_40980.npy'
    res = load_results(filepath)
    visualize_results(res, label='without adaptation', c='red')

    filepath = '../../../data/datasets/da_results_target_only_40980.npy'
    res = load_results(filepath)
    visualize_results(res, label='target_only', c='green')

    plt.ylabel('validation loss')
    plt.xlabel('number of target domain samples')
    plt.legend()
    plt.show()


    

    

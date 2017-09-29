"""
Visualization functions
"""
import argparse
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns

sys.path.append('../')
from prediction.batch.dataset_loaders import risk_dataset_loader

COLORS = ["red", "green", "blue", "purple", "yellow", "orange", "teal", 
    "black", "cyan", "magenta", "lightcoral", "darkseagreen"]

def maybe_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def get_dataset_feature_target_labels(datasets):

    for dataset in datasets:
        file = h5py.File(dataset['filepath'], 'r')
        feature_labels = file['risk'].attrs['feature_names']
        target_labels = file['risk'].attrs['target_names']
        file.close()
        break
    return feature_labels, target_labels

def compute_kl(p, q):
    p = np.asarray(p, dtype=np.float)
    p /= np.sum(p)
    q = np.asarray(q, dtype=np.float)
    q /= np.sum(q)

    kl = 0
    for (pv, qv) in zip(p, q):
        if pv != 0 and qv != 0:
            kl += pv * np.log(pv / qv)
    return kl

def compute_chi_sq(p, q):
    p = np.asarray(p, dtype=np.float)
    p /= np.sum(p)
    q = np.asarray(q, dtype=np.float)
    q /= np.sum(q)

    chi_sq = 0
    for (pv, qv) in zip(p, q):
        if pv != 0 or qv != 0:
            chi_sq += .5 * (pv - qv) ** 2 / (pv + qv)
    return chi_sq

def compare_feature_histograms(
        datasets, 
        output_directory, 
        nbins=50,
        function_name='compare_feature_histograms'):
    output_directory = os.path.join(output_directory, function_name)
    maybe_mkdir(output_directory)
    feature_labels, target_labels = get_dataset_feature_target_labels(datasets)
    num_features = len(feature_labels)
    total_kls = np.zeros(len(datasets) - 1)
    kls_list = [[] for _ in range(len(datasets) - 1)]
    for fidx in range(num_features):

        # track first hist b/c this is what the rest are compared with
        first_hist = None 
        for didx, dataset in enumerate(datasets):
            values = dataset['x_train'][:, fidx]
            if first_hist is None:
                first_hist, _ = np.histogram(values, bins=nbins)
                kl_value = 0.
            else:
                cur_hist, _ = np.histogram(values, bins=nbins)
                kl_value = compute_chi_sq(first_hist, cur_hist)
                total_kls[didx - 1] += kl_value
                kls_list[didx - 1] += [kl_value]

            print('feature {} {} / {}\tchisq={}'.format(feature_labels[fidx], fidx, num_features, kl_value))
            #plt.hist(values, nbins, color=COLORS[didx], alpha=.5, normed=True, 
            #         label='{}, chisq={:.2f}'.format(dataset['label'], kl_value))
            plt.hist(values, nbins, color=COLORS[didx], alpha=.5, normed=True, 
                     label='{}'.format(dataset['label']))
        plt.xlabel('Value of Feature')
        plt.ylabel('Frequency')
        plt.title('Histogram of {} values'.format(feature_labels[fidx]))
        plt.legend()
        output_filepath = os.path.join(output_directory, '{}'.format(feature_labels[fidx]))
        plt.savefig(output_filepath)
        plt.clf()

    # histogram of metric values by dataset
    for didx, kl_list in enumerate(kls_list):
        plt.hist(kl_list, 50, color=COLORS[didx], alpha=.5, normed=True, 
                label='{}, median chisq={:.2f}, mean chisq={:.2f}'.format(
                    dataset_labels[didx+1], 
                    np.median(kl_list),
                    np.mean(kl_list)))
    plt.xlabel('Chi Sq value')
    plt.ylabel('Fraction of Chi Sq values')
    plt.title('Histogram of Chi Sq values')
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15), 
        fancybox=True, 
        shadow=True)
    output_filepath = os.path.join(output_directory, 'kl')
    plt.savefig(output_filepath)
    plt.clf()

    idxs = reversed(np.argsort(kls_list[0]))
    for i in idxs:
        print(feature_labels[i])
        print("chi sq: {:.2f}".format(kls_list[0][i]))

    print([(dataset_labels[i+1],total_kls[i]) for i in range(len(datasets) - 1)])

def load_datasets(dataset_filepaths, debug_size):
    datasets = []
    for dataset_filepath in dataset_filepaths:
        dataset = risk_dataset_loader(
            dataset_filepath, 
            shuffle=False, 
            train_split=1., 
            debug_size=debug_size, 
            normalize=False, 
            timesteps=1,
            load_likelihood_weights=False)
        dataset['filepath'] = dataset_filepath
        dataset['label'] = os.path.split(dataset_filepath)[-1].replace('.h5', '')
        datasets.append(dataset)
    return datasets

if __name__ == '__main__':

    # parse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--function_names',
        nargs='+',  
        help="names of functions to run, space separated", 
        required=True)
    parser.add_argument('--dataset_filepaths', 
        nargs='+', 
        help='''space separated list of datasets to visualize.
            When these are compared to each other and there are multiple 
            ones, they are compared to the first dataset listed.''', 
        required=True)
    parser.add_argument('--output_directory', 
        default='', 
        type=str,
        help="directory where to save", 
        required=True)
    parser.add_argument('--debug_size', 
        default=100000, 
        type=int,
        help="max number of data points")
    args = parser.parse_args()

    # lookup the function
    _locals = locals()
    functions = []
    for function_name in args.function_names:
        if function_name not in _locals:
            raise ValueError('invalid function name: {}'.format(function_name))
        functions.append(_locals[function_name])

    # maybe make the output directory
    maybe_mkdir(args.output_directory)

    # load the datasets
    datasets = load_datasets(
        args.dataset_filepaths, 
        args.debug_size
    )

    for dataset in datasets:
        if 'bn' in dataset['label']:
            dataset['label'] = 'Original Data'
        if 'prediction' in dataset['label']:
            dataset['label'] = 'Generated Data'

    # run the functions
    for fn in functions:
        fn(datasets, args.output_directory)



import bisect
import collections
import csv
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import os 
from scipy.ndimage.filters import gaussian_filter
import scipy.stats
import sys
# import seaborn as sns

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset_loaders
import utils

COLORS = ["red", "green", "blue", "purple", "yellow", "orange", "teal", 
    "black", "cyan", "magenta", "lightcoral", "darkseagreen"]


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

def histogram_weights(dataset, output_filepath):
    weights = dataset['lw_train']
    std = np.std(weights)
    mean = np.mean(weights)
    weights[weights > mean + 3 * std] = 0.
    plt.hist(weights, 200, alpha=.5)
    plt.title('Likelihood Weights Histogram')
    plt.xlabel('weight value')
    plt.ylabel('count')
    plt.show()

def scatter_with_best_fit(features, targets, f_idx, t_idx, d_idx):
    """
    Description:
        - scatter a single set of features vs targets

    Args:
        - features: full set of features
        - targets: full set of targets
        - f_idx: index of feature
        - t_idx: index of target
    """
    # show line of best fit
    r, m = -1, -1
    if np.std(features[:, f_idx]) > 1e-8:

        (m, b, r, p_val, std_err) = scipy.stats.stats.linregress(
            features[:, f_idx], targets[:, t_idx])
        plt.plot(features[:, f_idx], m * features[:, f_idx] + b, 
            c=COLORS[d_idx], label='r: {:.4f} m: {:.4f}'.format(r, m))

    # scatter points
    plt.scatter(features[:, f_idx], targets[:, t_idx], alpha=.3)
    return r, m

def visualize_datasets(datasets, dataset_labels, feature_label, 
        target_labels, output_directory):
    """
    Description:
        - Visualize the features vs targets of a set of datasets.

    Args:
        - datasets: list of datasets
        - dataset_labels: names for datasets
        - feature_labels: names for features
        - target_labels: names for targets
        - output_directory: where to save the plots
    """

    # unpack values
    num_datasets = len(datasets)
    feature_sets = [d['x_train'] for d in datasets]
    target_sets = [d['y_train'] for d in datasets]
    num_features = feature_sets[0].shape[-1]
    num_targets = target_sets[0].shape[-1]

    # plot each feature against each target value
    # and also calculate the corresponding corr coeff
    coeff, slopes = [], []
    for f_idx in range(num_features):
        print("feature {}".format(f_idx))
        for t_idx in range(num_targets):
            fig = plt.figure(figsize=(10,10))
            lgds = []
            # plot each dataset as separate graph in figure
            for d_idx in range(num_datasets):

                plt.subplot('{}1{}'.format(num_datasets, int(d_idx + 1)))
                # plt.subplot(int('21{}'.format(d_idx + 1)))
                r, m = scatter_with_best_fit(feature_sets[d_idx], 
                    target_sets[d_idx], f_idx, t_idx, d_idx)
                coeff.append((r, feature_labels[f_idx], 
                    target_labels[t_idx]))
                slopes.append((r, feature_labels[f_idx], 
                    target_labels[t_idx]))
                
                plt.xlabel('{}'.format(feature_labels[f_idx]))
                plt.ylabel('{}'.format(target_labels[t_idx]))
                plt.title('{}: {} vs {}'.format(
                    dataset_labels[d_idx], 
                    feature_labels[f_idx], 
                    target_labels[t_idx]))
                l = plt.legend(bbox_to_anchor=(.8, -0.3), loc=8, borderaxespad=0.)
                
                # if did not scatter best fit line then there will not be 
                # any legend entry, so only add it if it was in fact plotted
                if r != -1 or m != -1:
                    lgds.append(l)

            # save the figure
            output_filepath = os.path.join(output_directory, 
                '{}_vs_{}.png'.format(
                feature_labels[f_idx], 
                target_labels[t_idx]))
            fig.savefig(output_filepath, bbox_extra_artists=lgds, 
                bbox_inches='tight')
            # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
            plt.savefig(output_filepath)
            plt.close()

    for t_idx in range(num_targets):
        fig = plt.figure(1)
        for d_idx in range(num_datasets):
            plt.subplot(int('31{}'.format(d_idx + 1)))
            n, bins, patches = plt.hist(
                target_sets[d_idx][:, t_idx], 50, alpha=0.5)
            plt.title('{}: {}'.format(
                    dataset_labels[d_idx], target_labels[t_idx]))
        output_filepath = os.path.join(
            output_directory, 'new_target_{}.png'.format(
                target_labels[t_idx]))
        print(output_filepath)
        fig.tight_layout()
        plt.savefig(output_filepath)
        plt.close()

    # sorting by correlation
    print(sorted(coeff))
    print(sorted(slopes))
    output_filepath = os.path.join(output_directory, 'corr.npz')
    np.savez(output_filepath, coeff=coeff, slopes=slopes)

def compute_feature_target_correlations(dataset):
    # unpack
    x, y = dataset['x_train'], dataset['y_train']
    num_samples, num_features = x.shape
    _, num_targets = y.shape

    # iterate through each feature
    coeffs = np.zeros((num_features, num_targets))
    for fidx in range(num_features):
        # take the coefficients corresponding to the targets
        coeffs[fidx, :] = np.corrcoef(x[:, fidx], y, rowvar=0)[0, 1:]
        if any(np.isnan(coeffs[fidx,:])):
            print(fidx)

    return coeffs

def collision_probability_by_behavior(dataset, politeness_idx=72):
    # output collision probability for the different behavior classes
    x, y = dataset['x_train'], dataset['y_train']
    tar_by_beh = collections.defaultdict(list)
    for s_idx in range(len(x)):
        politeness = x[s_idx, politeness_idx]
        target_probs = y[s_idx, :]
        tar_by_beh[politeness].append(target_probs)
    lines = [(p, np.mean(v, axis=0)) for (p,v) in tar_by_beh.items()]
    print("politeness (indicating behavior class) vs target means")
    for line in lines: print(line)

def report_poorly_performing_indices(idxs, data):
    batch_idxs = data['batch_idxs']
    seeds = data['seeds']
    for idx in idxs:
        # you could biject, but let's keep it simple
        for i, b in enumerate(batch_idxs):
            if b > idx:
                break
        seed = seeds[i]
        if i > 0:
            veh_idx = idx - batch_idxs[i - 1] + 1
        else:
            veh_idx = idx + 1
        print('seed: {}\tveh idx: {}'.format(seed, veh_idx))
        print('seed num veh: {}'.format(batch_idxs[i] - batch_idxs[i-1]))

def compare_dataset_targets_pairwise(d1, d2, target_labels):
    # truncate to the same number of samples
    y1, y2 = d1['y_train'], d2['y_train']
    num_samples = min(len(y1), len(y2))
    y1, y2 = y1[:num_samples], y2[:num_samples]

    # brier score by target
    brier = ((y1 - y2) ** 2).mean(axis=0)
    print('\nbrier scores: {}\n'.format(brier))

    # hypothesis test by target
    # null hypothesis: they are the same
    # alternative hypothesis: they are different
    y1_means, y2_means = y1.mean(axis=0), y2.mean(axis=0)
    p_hat = (y1_means + y2_means) / 2.
    z = np.abs((y1_means - y2_means) / np.sqrt(
        p_hat * (1 - p_hat) * (2 / num_samples)))
    z_alpha = 1.96
    for t, zt in enumerate(z):
        reject = False
        if zt > z_alpha:
            reject = True
        print('target: {}\t reject: {}\tz: {}'.format(
            target_labels[t], reject, zt))
    print('\n')

    # where did it perform poorly
    num_poor_samples = 50
    poor_idxs = np.argsort(np.sum(np.abs(y1 - y2), axis=1))
    for j, idx in enumerate(reversed(poor_idxs)):
        report_poorly_performing_indices([idx], d1)
        print('y1: {}\ny2: {}\n\n'.format(y1[idx], y2[idx]))
        if j > num_poor_samples:
            break

def sort_scenario_seeds_by_target(dataset, output_directory):
    # sort the scenario seeds by target probabilities
    x, y = dataset['x_train'], dataset['y_train']
    seeds, batch_idxs = dataset['seeds'], dataset['batch_idxs']

    # if already computed then just load
    metadata_filepath = os.path.join(output_directory, 'metadata.npz')
    if os.path.exists(metadata_filepath):
        metadata = np.load(metadata_filepath)
        sorted_seeds = metadata['sorted_seeds']
        sorted_means = metadata['sorted_means']
        sorted_bss = metadata['sorted_bss']
    # otherwise compute means by scenario batch and sort along with seeds
    else:
        mean_targets = []
        prev_bidx = 0
        for (i, bidx) in enumerate(batch_idxs):
            ys = y[prev_bidx:bidx]
            bs = bidx - prev_bidx
            # mean_seed = (tuple(np.mean(ys, axis=0)), seeds[i], bs)
            mean_seed = (np.mean(np.mean(ys, axis=0)), seeds[i], bs)
            mean_targets.append(mean_seed)
            prev_bidx = bidx

        sorted_means = sorted(mean_targets, reverse=True)
        sorted_seeds = [s for (_, s, _) in sorted_means]
        sorted_bss = [bs for (_, _, bs) in sorted_means]
        sorted_means = [m for (m, _, _) in sorted_means]

    # display some info
    print(sorted_seeds[:100])
    print(sorted_bss[:100])
    print(sorted_means[:10])

    plt.scatter(range(len(sorted_bss)), sorted_bss, alpha=.5)
    plt.show()
    plt.scatter(range(len(sorted_seeds)), sorted_seeds, alpha=.5)
    plt.show()
    original = sorted(list(zip(sorted_seeds, sorted_means)))
    means = np.array([m for (s,m) in original])
    seeds = [s for (s,m) in original]
    plt.scatter(seeds, means[:, 0], alpha=.5)
    plt.show()

    # save if have't already
    if not os.path.exists(metadata_filepath):
        np.savez(metadata_filepath, sorted_seeds=sorted_seeds, 
            sorted_means=sorted_means, sorted_bss=sorted_bss)

def compare_dataset_features(d1, d2, l1, l2):
    f1 = d1['x_train'][0]
    f2 = d2['x_train'][0]

    map_1 = {lv1:fv1 for (fv1,lv1) in zip(f1, l1)}
    map_2 = {lv2:fv2 for (fv2,lv2) in zip(f2, l2)}

    k1 = set(map_1.keys())
    k2 = set(map_2.keys())

    diff = k1.symmetric_difference(k2)
    print('different keys: {}\nlen: {}'.format(sorted(diff), len(diff)))

    same = k1.intersection(k2)
    for k in l2:
        if k in k1 and k in k2:
            print(k)
            print(map_1[k])
            print(map_2[k])
            input()

def report_high_prob_target_seeds_veh_idxs(data, output_filepath, 
        tidx=1, threshold=.5):
    # sort the targets
    batch_idxs = data['batch_idxs']
    seeds = data['seeds']
    targets = data['y_train'][:,tidx]
    idxs = list(reversed(np.argsort(targets)))

    # collect the output seeds and veh idxs
    rows = []
    for idx in idxs:
        # go through batch idxs until finding the scene of the vehicle
        i = bisect.bisect_right(batch_idxs, idx)

        # get the vehicle idx
        if i > 0:
            veh_idx = idx - batch_idxs[i - 1] + 1
        else:
            veh_idx = idx + 1

        # check if the target is above the threshold, if not break
        target_val = data['y_train'][idx,tidx]
        if target_val < threshold:
            break

        # write the seed and veh idx to file
        rows.append([seeds[i], veh_idx])

    outfile = open(output_filepath, 'w')
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(['seed','veh_index'])
    csv_writer.writerows(rows)
    outfile.close()

def visualize_features(dataset, feature_labels, dataset_label, output_directory):
    num_features = len(feature_labels)
    for i, fl in enumerate(feature_labels):
        print('feature {} {} / {}'.format(feature_labels[i], i, num_features))
        values = dataset['x_train'][:, i]
        plt.hist(values, 50, color='green', alpha=.5)
        plt.xlabel('Value of Feature')
        plt.ylabel('Count of Feature in Bin')
        plt.title('Histogram of {} values in dataset {}'.format(
            feature_labels[i], dataset_label))
        output_filepath = os.path.join(output_directory, 'hist_{}'.format(feature_labels[i]))
        plt.savefig(output_filepath)
        plt.clf()

def compare_feature_histograms(datasets, feature_labels, dataset_labels,
        output_directory, nbins=50):
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
            plt.hist(values, nbins, color=COLORS[didx], alpha=.5, normed=True, 
                label='{}, chisq={}'.format(dataset_labels[didx], kl_value))
        plt.xlabel('Value of Feature')
        plt.ylabel('Count of Feature in Bin')
        plt.title('Histogram of {} values'.format(feature_labels[fidx]))
        plt.legend()
        output_filepath = os.path.join(output_directory, 'hist_{}'.format(feature_labels[fidx]))
        plt.savefig(output_filepath)
        plt.clf()

    # histogram of kl values by dataset
    for didx, kl_list in enumerate(kls_list):
        plt.hist(kl_list, 50, color=COLORS[didx], alpha=.5, normed=True, 
                label='{}, median chisq={:.3f}, mean chisq={:.3f}'.format(
                    dataset_labels[didx+1], 
                    np.median(kl_list),
                    np.mean(kl_list)))
    plt.xlabel('Chi Sq value')
    plt.ylabel('Fraction of Chi Sq values')
    plt.title('Histogram of Chi Sq values')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, 
        shadow=True)
    output_filepath = os.path.join(output_directory, 'hist_kl')
    plt.savefig(output_filepath)
    plt.clf()

    idxs = reversed(np.argsort(kls_list[0]))
    for i in idxs:
        print(feature_labels[i])
        print("chi sq: {}".format(kls_list[0][i]))

    print([(dataset_labels[i+1],total_kls[i]) for i in range(len(datasets) - 1)])

def display_target_info(datasets, target_labels, dataset_labels, 
        output_directory):
    # basic stats
    for i, d in enumerate(datasets):
        num_samples = len(d['y_train'])
        means = np.mean(d['y_train'], axis=0)
        sums = np.sum(d['y_train'], axis=0)
        stds = np.std(d['y_train'], axis=0)
        line = "{}\n# samples: {}\ntarget means: {}\nsums: {}\nstds: {}\n".format(
            dataset_labels[i], num_samples, means, sums, stds)
        print(line)

    # correlations
    print('target correlation matrix: ')
    print(np.corrcoef(datasets[-1]['y_train'], rowvar=0))

    means = [np.mean(d['y_train'], axis=0) for d in datasets]
    means = np.array(means)
    num_datasets, num_targets = means.shape
    for tidx in range(num_targets):
        output_filepath = os.path.join(output_directory, 'mean_{}.png'.format(target_labels[tidx]))
        plt.scatter(range(num_datasets), means[:,tidx],label=target_labels[tidx])
        plt.title('{} vs iterations'.format(target_labels[tidx]))
        plt.xlabel('iterations')
        plt.ylabel('pr({})'.format(target_labels[tidx]))
        plt.savefig(output_filepath)
        plt.clf()


    means = means / np.max(means, axis=0)
    num_datasets, target_dim = means.shape
    width = 0.1
    rects = []
    for tidx in range(target_dim):
        fig, ax = plt.subplots()
        size = max(6, int(num_datasets / 1.5))
        fig.set_size_inches(size, size)
        color = np.array([.1,.1,.1])
        color_step = (1 - color) / num_datasets - 1e-4
        for didx in range(num_datasets):
            color += color_step
            rect = ax.bar(width * didx, means[didx,tidx], width, color=tuple(color), 
                alpha=.9, edgecolor="black")
            rects.append(rect)
        ax.set_ylabel('pr({})'.format(target_labels[tidx]))
        ax.set_title('pr({}) across datasets'.format(target_labels[tidx]))
        ax.legend(([rect[0] for rect in rects]), dataset_labels, bbox_to_anchor=(.5, -1.1))
        output_filepath = os.path.join(output_directory, 'comparison_{}.png'.format(target_labels[tidx]))
        plt.savefig(output_filepath)
        plt.clf()

    # histogram of target values
    _, num_targets = datasets[0]['y_train'].shape
    num_datasets = len(datasets)
    num_cols = 1
    num_rows = int(np.ceil(num_datasets / num_cols))
    target_sets = [d['y_train'] for d in datasets]
    # fig = plt.figure(figsize=(num_rows, num_cols))
    fig.set_size_inches(num_cols * 4, num_rows * 2)
    ignore_zeros = True
    for t_idx in range(num_targets):
        for d_idx in range(num_datasets):
            plt.subplot(num_rows, num_cols, d_idx + 1)
            if ignore_zeros:
                n, bins, patches = plt.hist(
                    target_sets[d_idx][:, t_idx][target_sets[d_idx][:, t_idx] > 0], 
                    50, alpha=0.5)
            else:
                n, bins, patches = plt.hist(target_sets[d_idx][:, t_idx], 50, alpha=0.5)
            plt.title('{}: {}'.format(
                    dataset_labels[d_idx], target_labels[t_idx]))
            plt.ylabel('{}'.format(target_labels[t_idx]))
        output_filepath = os.path.join(
            output_directory, 'hist_{}.png'.format(
                target_labels[t_idx]))
        print(output_filepath)
        # plt.tight_layout()
        plt.savefig(output_filepath)
        plt.clf()

if __name__ == '__main__':
    # labels stored in external csv files 
    feature_labels_filepath = '../../data/datasets/features.csv'
    target_labels_filepath = '../../data/datasets/targets.csv'

    # load in labels
    feature_labels = utils.load_labels(feature_labels_filepath)
    target_labels = utils.load_labels(target_labels_filepath)

    ## the dataset filepaths to visualize along with labels
    input_filepaths = [
        # '../../data/datasets/may/proposal_five_sec_bn.h5',
        '../../data/datasets/may/ngsim_5_sec_10_timestep.h5',
        '../../data/datasets/may/bn_aug_5_sec_10_timestep.h5',
        '../../data/datasets/may/additional_5_sec_10_timestep.h5',
        # '../../data/datasets/may/safe_additional_5_sec_10_timestep.h5'
        # '../../data/datasets/april/risk_20_sec_large.h5',
        # '../../data/datasets/april/risk_mc_1.h5',
        # '../../data/datasets/april/risk_mc_2.h5',
        # '../../data/datasets/april/risk_mc_4.h5',
        # '../../data/datasets/april/risk_mc_8.h5',
        # '../../data/datasets/april/risk_mc_16.h5',
        # '../../data/datasets/april/risk_mc_32.h5'
        # '../../data/datasets/bootstrap_discount/iter_99.h5',
        # '../../data/datasets/bootstrap_mc_compare.h5',
        # '../../data/datasets/march/risk_20_sec_3_timesteps.h5',
    ]
    dataset_labels = [
        # 'proposal_five_seconds',
        'ngsim_5_second',
        'bayesnet',
        'heuristic',
        # 'safe_heuristic'
        # '20_sec_large'
        # 'mc_1',
        # 'mc_2',
        # 'mc_4',
        # 'mc_8',
        # 'mc_16',
        # 'mc_32'
        # 'boot',
        # 'compare',
        # 'risk_5'
    ]

    # num_iters = 100
    # input_filenames = ['iter_{}.h5'.format(i) for i in range(num_iters)]
    # basedir = '../../data/datasets/bootstrap_discount'
    # input_filepaths = [os.path.join(basedir, f) for f in input_filenames]
    # dataset_labels = ['{}_seconds'.format(i) for i in range(1, num_iters + 1)]
    
    # # check for / create output directory
    base_directory = '../../data/visualizations/'
    output_directory = os.path.join(
        base_directory, os.path.split(
            input_filepaths[0])[-1]).replace('.h5', '')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # load in each dataset
    load_likelihood_weights = False
    debug_size = 500000
    datasets = [dataset_loaders.risk_dataset_loader(
        input_filepath, shuffle=False, train_split=1., 
        debug_size=debug_size, normalize=False, timesteps=1,
        load_likelihood_weights=load_likelihood_weights) for 
        input_filepath in input_filepaths]

    feature_labels = h5py.File(input_filepaths[-1], 'r')['risk'].attrs['feature_names']

    # display basic info about the targets
    display_target_info(datasets, target_labels, dataset_labels, output_directory)

    # # histogram the features
    # visualize_features(datasets[-1], feature_labels, dataset_labels[-1],
    #     output_directory)

    # compare histogram of features across datasets
    compare_feature_histograms(datasets, feature_labels, dataset_labels,
        output_directory)

    # ## analyze behavior
    # for i, dataset in enumerate(datasets):
    #     print(dataset_labels[i])
    #     collision_probability_by_behavior(dataset)

    # # compute feature target correlations and write to file
    # coeffs = compute_feature_target_correlations(datasets[-1])
    # output_filepath = os.path.join(output_directory, 'coeff.csv')
    # feature_names = h5py.File(input_filepaths[-1], 'r')['risk'].attrs['feature_names']
    # utils.write_to_csv(output_filepath, coeffs, feature_names, target_labels)

    # # visualize the datasets
    # visualize_datasets(datasets, dataset_labels, feature_labels, 
    #     target_labels, output_directory)

    # output_directory = os.path.join(base_directory, dataset_labels[-1])
    # sort_scenario_seeds_by_target(datasets[-1], output_directory)

    # tidx = 0
    # output_filepath = os.path.join(
    #     output_directory, 'seed_veh_idx_target_{}.csv'.format(tidx))
    # report_high_prob_target_seeds_veh_idxs(
    #     datasets[0], output_filepath, tidx=tidx)

    # # compare two datasets
    # compare_dataset_targets_pairwise(
    #     datasets[0], datasets[1], target_labels)

    # feature_labels_filepath = '../../data/datasets/features_multi.csv'
    # feature_labels_multi = utils.load_labels(feature_labels_filepath)
    # compare_dataset_features(datasets[0], datasets[1], feature_labels_multi, feature_labels)

    # hist weights
    # output_filepath = os.path.join(output_directory, 'sample_weights.png')
    # histogram_weights(datasets[0], output_filepath)


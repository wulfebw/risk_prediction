
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os 
import seaborn as sns
import sklearn.metrics

LABELS = ['base', 'behavior', 'neighbor', 'all']

def compare_prc_curves(input_filepaths, output_filepath):
    for i, input_filepath in enumerate(input_filepaths):
        d = np.load(input_filepath)
        precision, recall = d['precision'], d['recall']
        avg_precision = float(d['avg_precision'])
        plt.plot(recall, precision, label='{} (area = {:.3f})'.format(LABELS[i], avg_precision))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Across Feature Sets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()

if __name__ == '__main__':
    base_dir = '/Users/wulfebw/Desktop/5_sec_compression_figs'
    run_dirs = ['test_base', 'test_base_beh', 'test_base_beh_neigh', 
        'test_base_beh_neigh_beh']
    filename = 'stats_validation.npz'
    input_filepaths = [os.path.join(base_dir, run_dir, filename) 
        for run_dir in run_dirs]
    output_filepath = '/Users/wulfebw/Desktop/5_sec_compression_figs/summary.png'
    compare_prc_curves(input_filepaths, output_filepath)
import argparse
import configparser
import os
import subprocess

from utils import maybe_mkdir, build_cmd, run_cmd, print_intro

ROOTDIR = os.path.abspath(os.path.join('../'))


# setup
def run_setup(config):
    expdir = config.defaults().get('expdir')
    maybe_mkdir(expdir)
    maybe_mkdir(os.path.join(expdir, 'log'))
    maybe_mkdir(os.path.join(expdir, 'data'))
    maybe_mkdir(os.path.join(expdir, 'data', 'snapshots'))
    maybe_mkdir(os.path.join(expdir, 'data', 'summaries'))
    maybe_mkdir(os.path.join(expdir, 'viz'))

# collection
def run_ngsim_collection(config):
    pass

def run_heuristic_collection(config):
    s = 'collection'
    cmd = 'julia -p {} '.format(config.get(s, 'nprocs'))
    cmd += 'run_collect_dataset.jl '
    cmd += build_cmd(config.items(s), prefix='col/')
    cmd_dir = os.path.join(ROOTDIR, 'collection')
    run_cmd(cmd, config.get(s, 'logfile'), cmd_dir=cmd_dir, 
        dry_run=config.dry_run)
    
def run_collection(config):
    s = 'collection'
    print_intro(s)
    
    data_source = config.get(s, 'data_source')
    if data_source == 'heuristic':
        run_heuristic_collection(config)
    elif data_source == 'ngsim':
        run_ngsim_collection(config)

# generation
def fit_bayes_net(config):
    s = 'generation' 
    cmd = 'julia run_fit_bayes_net.jl '
    cmd += '--input_filepath {} '.format(
        config.get('collection', 'col/output_filepath'))
    cmd += '--output_filepath {} '.format(
        config.get(s, 'base_bn_filepath'))
    cmd_dir = os.path.join(ROOTDIR, 'scene_generation')
    run_cmd(cmd, config.get(s, 'base_bn_logfile'), cmd_dir=cmd_dir, 
            dry_run=config.dry_run)

def fit_proposal_bayes_net(config):
    s = 'generation'
    cmd = 'julia -p {} run_fit_proposal_bayes_net.jl '.format(config.get(s, 'nprocs'))
    # want to use the same configuration for fitting the proposal 
    # BN as will be used for data generation, so add the data gen flags
    # first, and then add the explictly proposal-BN training flags second
    # this lets the explict ones override, but uses the data gen flags 
    # as defaults
    cmd += build_cmd(config.items(s), prefix='gen/')
    cmd += '--dataset_filepath {} '.format(
        config.get('collection', 'col/output_filepath'))
    cmd += '--base_bn_filepath {} '.format(
        config.get(s, 'base_bn_filepath'))
    cmd += '--output_filepath {} '.format(
        config.get(s, 'prop_bn_filepath'))
    cmd += build_cmd(config.items(s), prefix='prop/')
    cmd_dir = os.path.join(ROOTDIR, 'scene_generation')
    run_cmd(cmd, config.get(s, 'prop_bn_logfile'), cmd_dir=cmd_dir, 
        dry_run=config.dry_run)

def generate_prediction_data(config):
    s = 'generation'
    cmd = 'julia -p {} run_collect_dataset.jl '.format(config.get(s, 'nprocs'))
    cmd += '--base_bn_filepath {} '.format(
        config.get(s, 'base_bn_filepath'))
    cmd += '--prop_bn_filepath {} '.format(
        config.get(s, 'prop_bn_filepath'))
    cmd += build_cmd(config.items(s), prefix='gen/')
    cmd_dir = os.path.join(ROOTDIR, 'collection')
    run_cmd(cmd, config.get(s, 'generation_logfile'), cmd_dir=cmd_dir, 
        dry_run=config.dry_run)

def subselect_prediction_data(config):
    s = 'generation'
    cmd = 'python subselect_dataset.py '
    cmd += '--dataset_filepath {} '.format(config.get(s, 'gen/output_filepath'))
    cmd += '--subselect_filepath {} '.format(
        config.get(s, 'subselect_dataset'))
    cmd += '--subselect_feature_filepath {} '.format(
        config.get(s, 'subselect_feature_dataset'))
    cmd += '--subselect_proposal_filepath {} '.format(
        config.get(s, 'subselect_proposal_dataset'))
    cmd_dir = os.path.join(ROOTDIR, 'collection')
    run_cmd(cmd, config.get(s, 'subselect_logfile'), cmd_dir=cmd_dir, 
        dry_run=config.dry_run)

def run_generation(config):
    s = 'generation'
    print_intro(s)

    fit_bayes_net(config)
    fit_proposal_bayes_net(config)
    generate_prediction_data(config)
    subselect_prediction_data(config)

# prediction
def run_batch_prediction(config):
    s = 'prediction'
    cmd = 'python fit_predictor.py '
    cmd += build_cmd(config.items(s), prefix='batch/')
    cmd_dir = os.path.join(ROOTDIR, 'prediction/batch')
    run_cmd(cmd, config.get(s, 'logfile'), cmd_dir=cmd_dir, dry_run=config.dry_run)

def run_td_prediction(config):
    s = 'prediction'
    cmd = 'python train.py '
    cmd += build_cmd(config.items(s), prefix='td/')
    cmd_dir = os.path.join(ROOTDIR, 'prediction/td/scripts/training')
    run_cmd(cmd, config.get(s, 'logfile'), cmd_dir=cmd_dir, dry_run=config.dry_run)
    print('Async prediction running in the background...')
    print("Enter 'tmux attach -t a3c' to attach")
    print("Enter 'tmux kill-session -t a3c' to end training manually")

def run_prediction(config):
    s = 'prediction'
    print_intro(s)

    prediction_type = config.get(s, 'prediction_type')
    if prediction_type == 'batch':
        run_batch_prediction(config)
    elif prediction_type == 'td':
        run_td_prediction(config)

def run_experiment(config):
    run_setup(config)
    run_collection(config)
    run_generation(config)
    run_prediction(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parses the config filepath')
    parser.add_argument('-c', '--config_filepath', type=str, 
        help='config filepath.', default='../../data/configs/test.cfg')
    parser.add_argument('-d', '--dry_run', action='store_true', 
        help='print cmds but not run them', default=False)
    args = parser.parse_args()
    config = configparser.SafeConfigParser()
    config.dry_run = args.dry_run
    config.read(args.config_filepath)
    run_experiment(config)

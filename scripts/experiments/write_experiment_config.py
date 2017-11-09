import configparser
import os

# constant accross all
EXPERIMENT_NAME = 'debug'                #
DEFAULTS = {
    'nprocs': 25,                                                       #
    'expdir': '/scratch/wulfebw/experiments/{}'.format(EXPERIMENT_NAME),
    #'expdir': '../../data/experiments/{}'.format(EXPERIMENT_NAME),
    'num_lanes': 1,
    'err_p_a_to_i': .05,
    'err_p_i_to_a': .3,
    'overall_response_time': .2,
    'lon_accel_std_dev': .3,                                        #
    'lat_accel_std_dev': .1,                                        #
}

def write_collection(config):
    s = 'collection'
    config.add_section(s)
    
    # logistics
    config.set(s, 'logfile', '%(expdir)s/log/collection.log')
    config.set(s, 'data_source', 'heuristic')

    # common
    config.set(s, 'col/output_filepath', '%(expdir)s/data/bn_train_data.h5')

    ## feature extraction
    config.set(s, 'col/feature_timesteps', '1')
    config.set(s, 'col/feature_step_size', '1')
    config.set(s, 'col/extractor_type', 'multi')
    config.set(s, 'col/extract_core', 'true')
    config.set(s, 'col/extract_temporal', 'true')
    config.set(s, 'col/extract_well_behaved', 'true')
    config.set(s, 'col/extract_neighbor', 'true')
    config.set(s, 'col/extract_behavioral', 'true')
    config.set(s, 'col/extract_neighbor_behavioral', 'true')
    config.set(s, 'col/extract_car_lidar', 'false')
    config.set(s, 'col/extract_car_lidar_range_rate', 'false')
    config.set(s, 'col/extract_road_lidar', 'false')

    # heuristic collection
    config.set(s, 'col/generator_type', 'factored')
    config.set(s, 'col/num_lanes', '%(num_lanes)s')
    config.set(s, 'col/num_scenarios', '1500')                             #
    config.set(s, 'col/num_monte_carlo_runs', '100')                        #
    config.set(s, 'col/err_p_a_to_i', '%(err_p_a_to_i)s')
    config.set(s, 'col/err_p_i_to_a', '%(err_p_i_to_a)s')
    config.set(s, 'col/overall_response_time', '%(overall_response_time)s')
    config.set(s, 'col/lon_accel_std_dev', '%(lon_accel_std_dev)s')
    config.set(s, 'col/lat_accel_std_dev', '%(lat_accel_std_dev)s')
    config.set(s, 'col/prime_time', '60.')
    config.set(s, 'col/sampling_time', '20.')
    config.set(s, 'col/min_init_dist', '10')                                #
    config.set(s, 'col/heuristic_behavior_type', 'correlated')                 #
    # don't change these numbers of vehicles
    config.set(s, 'col/max_num_vehicles', '70') 
    config.set(s, 'col/min_num_vehicles', '70')

    # ngsim collection
    # TODO

    # subselect
    config.set(s, 'subselect_logfile', '%(expdir)s/log/col_subselect.log')
    config.set(s, 'subselect_feature_dataset', 
        '%(expdir)s/data/subselect_feature_validation_data.h5')
    config.set(s, 'subselect_proposal', 'False')
    
def write_generation(config):
    s = 'generation'
    config.add_section(s)

    # logistics
    config.set(s, 'base_bn_logfile', '%(expdir)s/log/base_bn.log')
    config.set(s, 'prop_bn_logfile', '%(expdir)s/log/prop_bn.log')
    config.set(s, 'generation_logfile', '%(expdir)s/log/generation.log')
    config.set(s, 'subselect_logfile', '%(expdir)s/log/gen_subselect.log')

    # common 
    feature_timesteps = int(config.get('collection', 'col/feature_timesteps'))
    feature_step_size = int(config.get('collection', 'col/feature_step_size'))

    # base bayes net training
    config.set(s, 'base_bn_filepath', '%(expdir)s/data/base_bn.jld')

    # proposal bayes net training
    # note: use the dataset generation parameters defined below
    # e.g., the sampling time from generation
    # and do this by passing them in explictly in the run call
    config.set(s, 'prop_bn_filepath', '%(expdir)s/data/prop_bn.jld')
    config.set(s, 'prop/num_monte_carlo_runs', '2')                     #
    config.set(s, 'prop/cem_end_prob', '.1')
    config.set(s, 'prop/max_iters', '1000')                             #
    config.set(s, 'prop/population_size', '10000')                       #
    config.set(s, 'prop/top_k_fraction', '.2')
    config.set(s, 'prop/n_prior_samples', '20000')
    config.set(s, 'prop/n_static_prior_samples', '4000')
    config.set(s, 'prop/viz_dir', '%(expdir)s/viz/prop')
    config.set(s, 'prop/start_target_timestep', '101')
    config.set(s, 'prop/end_target_timestep', '200')
    # prime time for proposal should be exactly the number of feature timesteps

    # generation of validation / training data
    config.set(s, 'gen/output_filepath', '%(expdir)s/data/prediction_data.h5')

    ## feature extraction
    config.set(s, 'gen/feature_timesteps', '{}'.format(feature_timesteps))
    config.set(s, 'gen/feature_step_size', '{}'.format(feature_step_size))
    config.set(s, 'gen/extractor_type', 'multi')
    config.set(s, 'gen/extract_core', 'true')
    config.set(s, 'gen/extract_temporal', 'true')
    config.set(s, 'gen/extract_well_behaved', 'true')
    config.set(s, 'gen/extract_neighbor', 'true')
    config.set(s, 'gen/extract_behavioral', 'true')
    config.set(s, 'gen/extract_neighbor_behavioral', 'true')
    config.set(s, 'gen/extract_car_lidar', 'false')
    config.set(s, 'gen/extract_car_lidar_range_rate', 'false')
    config.set(s, 'gen/extract_road_lidar', 'false')

    ## collection with bayes net
    config.set(s, 'gen/generator_type', 'joint')
    config.set(s, 'gen/num_scenarios', '10000')                          #
    config.set(s, 'gen/num_monte_carlo_runs', '100')                        #
    config.set(s, 'gen/num_lanes', '%(num_lanes)s')
    config.set(s, 'gen/err_p_a_to_i', '%(err_p_a_to_i)s')
    config.set(s, 'gen/err_p_i_to_a', '%(err_p_i_to_a)s')
    config.set(s, 'gen/overall_response_time', '%(overall_response_time)s')
    config.set(s, 'gen/lon_accel_std_dev', '%(lon_accel_std_dev)s')
    config.set(s, 'gen/lat_accel_std_dev', '%(lat_accel_std_dev)s')
    prime_time = (feature_timesteps * feature_step_size) * .1 + .4
    config.set(s, 'gen/prime_time', '{}'.format(prime_time))
    config.set(s, 'gen/sampling_time', '20.')                               #
    config.set(s, 'gen/heuristic_behavior_type', 'correlated')                 #
    config.set(s, 'gen/max_num_vehicles', '70')                            #
    config.set(s, 'gen/min_num_vehicles', '70')                            #

    # subselect dataset filepath
    config.set(s, 'subselect_dataset', 
        '%(expdir)s/data/subselect_prediction_data.h5')
    config.set(s, 'subselect_feature_dataset', 
        '%(expdir)s/data/subselect_feature_prediction_data.h5')
    config.set(s, 'subselect_proposal_dataset', 
        '%(expdir)s/data/subselect_proposal_prediction_data.h5')
    config.set(s, 'subselect_proposal', 'True')

def write_visualization(config):
    s = 'visualization'
    config.add_section(s)

    # logistics
    config.set(s, 'logfile', '%(expdir)s/log/visualization.log')

    # datasets to visualize
    dataset_filepaths = ' '.join([
        config.get('collection', 'col/output_filepath'),
        config.get('generation', 'subselect_proposal_dataset')
    ])
    config.set(s, 'viz/dataset_filepaths', dataset_filepaths)

    # functions to call
    config.set(s, 'viz/function_names', 'compare_feature_histograms')

    # where to put it and how much to do
    config.set(s, 'viz/output_directory', '%(expdir)s/viz/viz')
    config.set(s, 'viz/debug_size', '100000')

def write_prediction(config):
    s = 'prediction'
    config.add_section(s)

    # logistics
    config.set(s, 'prediction_type', 'batch')
    config.set(s, 'logfile', '%(expdir)s/log/prediction.log')

    # td prediction
    ## logistics
    config.set(s, 'td/log-dir', '%(expdir)s/data/')
    config.set(s, 'td/num-workers', '%(nprocs)s')
    config.set(s, 'td/config', 'risk_env_config')

    ### validation dataset
    config.set(s, 'td/validation_dataset_filepath', config.get('generation', 'subselect_proposal_dataset'))
    ### bayes net filepaths
    config.set(s, 'td/base_bn_filepath', config.get('generation', 'base_bn_filepath')) 
    config.set(s, 'td/prop_bn_filepath', config.get('generation', 'prop_bn_filepath'))

    ## viz
    config.set(s, 'td/viz_dir', config.get('generation', 'prop/viz_dir'))

    ## hyperparams
    config.set(s, 'td/hidden_layer_sizes', "'128,128,128'")                   #
    config.set(s, 'td/local_steps_per_update', '100')               #
    config.set(s, 'td/learning_rate', '5e-4')
    config.set(s, 'td/learning_rate_end', '5e-5')
    config.set(s, 'td/dropout_keep_prob', '1.')
    config.set(s, 'td/l2_reg', '0.')
    config.set(s, 'td/target_loss_index', '2')                       #
    horizon = float(config.get('generation', 'gen/sampling_time')) / .1
    discount = (horizon - 1) / horizon
    config.set(s, 'td/discount', str(discount))
    config.set(s, 'td/n_global_steps', '10000000')                   #
    config.set(s, 'td/max_timesteps', '1000')                         #
    config.set(s, 'td/prime_time', '0.')

    # batch prediction
    ## dataset / logistics
    config.set(s, 'batch/dataset_filepath', config.get(
        'generation', 'subselect_proposal_dataset'))
    config.set(s, 'batch/snapshot_dir', '%(expdir)s/data/snapshots')
    config.set(s, 'batch/viz_dir', '%(expdir)s/viz/prediction')
    config.set(s, 'batch/summary_dir', '%(expdir)s/data/summaries')

    ## hyperparams
    config.set(s, 'batch/batch_size', '2000')                              #
    config.set(s, 'batch/num_epochs', '200')
    config.set(s, 'batch/save_every', '2')
    config.set(s, 'batch/debug_size', '10000000')                            #
    config.set(s, 'batch/target_index', '2')                              #
    config.set(s, 'batch/hidden_layer_dims', "'128 128 128'")
    config.set(s, 'batch/learning_rate', '5e-4')
    config.set(s, 'batch/min_lr', '1e-5')
    config.set(s, 'batch/decrease_lr_threshold', '.0')
    config.set(s, 'batch/decay_lr_ratio', '.995')
    config.set(s, 'batch/loss_type', 'ce')                              #
    config.set(s, 'batch/task_type', 'regression')                      #
    num_target_bins = None
    if num_target_bins is not None:
        config.set(s, 'batch/num_target_bins', num_target_bins)              #
    config.set(s, 'batch/dropout_keep_prob', '.5')
    config.set(s, 'batch/use_batch_norm', 'True')
    config.set(s, 'batch/l2_reg', '1e-5')
    config.set(s, 'batch/timesteps', config.get(
        'generation', 'gen/feature_timesteps'))
    config.set(s, 'batch/use_likelihood_weights', 'True')

def write_validation(config):
    s = 'validation'
    config.add_section(s)

    # prediction on the original dataset used for collection
    # using the network trained on the generated data
    config.set(s, 'indirect/logfile', '%(expdir)s/log/indirect_validation.log')
    config.set(s, 'indirect/dataset_filepath', config.get(
        'collection', 'col/output_filepath'))
    config.set(s, 'indirect/summary_dir', '%(expdir)s/data/indirect_validation_summaries')
    config.set(s, 'indirect/viz_dir', '%(expdir)s/viz/indirect_validation')
    config.set(s, 'indirect/load_network', 'True')
    config.set(s, 'indirect/num_epochs', '0')
    config.set(s, 'indirect/learning_rate', '0')
    config.set(s, 'indirect/min_lr', '0')
    config.set(s, 'indirect/use_likelihood_weights', 'False')
    config.set(s, 'indirect/train_split', '.01')

    # train and predict on the original dataset directly, but only using the 
    # 0/1 labels
    config.set(s, 'direct/logfile', '%(expdir)s/log/direct_validation.log')
    config.set(s, 'direct/dataset_filepath', config.get(
        'collection', 'col/output_filepath'))
    config.set(s, 'direct/summary_dir', '%(expdir)s/data/direct_validation_summaries')
    config.set(s, 'direct/viz_dir', '%(expdir)s/viz/direct_validation')
    config.set(s, 'direct/snapshot_dir', '%(expdir)s/data/direct_snapshots')
    config.set(s, 'direct/use_likelihood_weights', 'False')
    config.set(s, 'direct/num_target_bins', '2')

def write_config(filepath):
    config = configparser.SafeConfigParser(defaults=DEFAULTS)
    write_collection(config)
    write_generation(config)
    write_visualization(config)
    write_prediction(config)
    write_validation(config)
    with open(filepath, 'w') as outfile:
        config.write(outfile)

if __name__ == '__main__':
    
    filepath = '../../data/configs/{}.cfg'.format(EXPERIMENT_NAME)
    write_config(filepath)

import configparser

# constant accross all
DEFAULTS = {
    'nprocs': 1,
    'expdir': '../../data/experiments/test',
    'num_lanes': 1,
    'err_p_a_to_i': .01,
    'err_p_i_to_a': .3,
    'overall_response_time': .2,
    'lon_accel_std_dev': 1.,
    'lat_accel_std_dev': .2,
}

def write_collection(config):
    s = 'collection'
    config.add_section(s)
    
    # logistics
    config.set(s, 'logfile', '%(expdir)s/log/collection.log')
    config.set(s, 'data_source', 'heuristic')

    # common
    config.set(s, 'output_filepath', '%(expdir)s/data/bn_train_data.h5')

    ## feature extraction
    config.set(s, 'col/feature_timesteps', '1')
    config.set(s, 'col/extractor_type', 'multi')
    config.set(s, 'col/extract_core', 'true')
    config.set(s, 'col/extract_temporal', 'true')
    config.set(s, 'col/extract_well_behaved', 'true')
    config.set(s, 'col/extract_neighbor', 'true')
    config.set(s, 'col/extract_behavioral', 'false')
    config.set(s, 'col/extract_neighbor_behavioral', 'false')
    config.set(s, 'col/extract_car_lidar', 'true')
    config.set(s, 'col/extract_car_lidar_range_rate', 'true')
    config.set(s, 'col/extract_road_lidar', 'false')

    # heuristic collection
    config.set(s, 'col/generator_type', 'factored')
    config.set(s, 'col/num_lanes', '%(num_lanes)s')
    config.set(s, 'col/num_scenarios', '1')
    config.set(s, 'col/num_monte_carlo_runs', '1')
    config.set(s, 'col/err_p_a_to_i', '%(err_p_a_to_i)s')
    config.set(s, 'col/err_p_i_to_a', '%(err_p_i_to_a)s')
    config.set(s, 'col/overall_response_time', '%(overall_response_time)s')
    config.set(s, 'col/lon_accel_std_dev', '%(lon_accel_std_dev)s')
    config.set(s, 'col/lat_accel_std_dev', '%(lat_accel_std_dev)s')
    config.set(s, 'col/prime_time', '30.')
    config.set(s, 'col/sampling_time', '.1')
    config.set(s, 'col/max_num_vehicles', '50')
    config.set(s, 'col/min_num_vehicles', '50')

    # ngsim collection
    # TODO
    
def write_generation(config):
    s = 'generation'
    config.add_section(s)

    # logistics
    config.set(s, 'logfile', '%(expdir)s/log/generation.log')

    # base bayes net training
    config.set(s, 'base_bn_filepath', '%(expdir)s/data/base_bn.jld')

    # proposal bayes net training
    config.set(s, 'prop_bn_filepath', '%(expdir)s/data/prop_bn.jld')
    config.set(s, 'prop/num_monte_carlo_runs', '1')
    config.set(s, 'prop/prime_time', '0.')
    config.set(s, 'prop/sampling_time', '1.')
    config.set(s, 'prop/cem_end_prob', '.5')
    config.set(s, 'prop/max_iters', '1')
    config.set(s, 'prop/population_size', '10')
    config.set(s, 'prop/top_k_fraction', '.5')
    config.set(s, 'prop/n_prior_samples', '50000')

    # generation of validation / training data
    ## feature extraction
    feature_timesteps = 10
    config.set(s, 'gen/feature_timesteps', '{}'.format(feature_timesteps))
    feature_step_size = 1
    config.set(s, 'gen/feature_step_size', '{}'.format(feature_step_size))
    config.set(s, 'gen/extractor_type', 'multi')
    config.set(s, 'gen/extract_core', 'true')
    config.set(s, 'gen/extract_temporal', 'true')
    config.set(s, 'gen/extract_well_behaved', 'true')
    config.set(s, 'gen/extract_neighbor', 'false')
    config.set(s, 'gen/extract_behavioral', 'false')
    config.set(s, 'gen/extract_neighbor_behavioral', 'false')
    config.set(s, 'gen/extract_car_lidar', 'true')
    config.set(s, 'gen/extract_car_lidar_range_rate', 'true')
    config.set(s, 'gen/extract_road_lidar', 'false')

    ## collection with bayes net
    config.set(s, 'gen/generator_type', 'joint')
    config.set(s, 'gen/num_lanes', '%(num_lanes)s')
    config.set(s, 'gen/num_scenarios', '1')
    config.set(s, 'gen/num_monte_carlo_runs', '1')
    config.set(s, 'gen/err_p_a_to_i', '%(err_p_a_to_i)s')
    config.set(s, 'gen/err_p_i_to_a', '%(err_p_i_to_a)s')
    config.set(s, 'gen/overall_response_time', '%(overall_response_time)s')
    config.set(s, 'gen/lon_accel_std_dev', '%(lon_accel_std_dev)s')
    config.set(s, 'gen/lat_accel_std_dev', '%(lat_accel_std_dev)s')
    prime_time = (feature_timesteps * feature_step_size) * .1 + .2
    config.set(s, 'gen/prime_time', '{}'.format(prime_time))
    config.set(s, 'gen/sampling_time', '.1')
    config.set(s, 'gen/max_num_vehicles', '50')
    config.set(s, 'gen/min_num_vehicles', '50')

def write_prediction(config):
    s = 'prediction'
    config.add_section(s)

    # logistics
    config.set(s, 'prediction_type', 'batch')

def write_config(filepath):
    config = configparser.SafeConfigParser(defaults=DEFAULTS)
    write_collection(config)
    write_generation(config)
    write_prediction(config)
    with open(filepath, 'w') as outfile:
        config.write(outfile)

if __name__ == '__main__':
    filepath = '../../data/configs/test.cfg'
    write_config(filepath)

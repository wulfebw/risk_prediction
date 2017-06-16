
class Config(object):
    def __init__(self):

        # environment options
        self.julia_env_id = 'BayesNetRiskEnv-v0'

        ## generation
        self.num_lanes = 1
        self.roadway_type = 'stadium'
        self.max_num_vehicles = 100
        self.base_bn_filepath = "../../data/bayesnets/base_test.jld"
        self.prop_bn_filepath = "../../data/bayesnets/prop_test.jld"
        self.lon_accel_std_dev = 1.
        self.lat_accel_std_dev = .1
        self.overall_response_time = .2
        self.lon_response_time = .2
        self.err_p_a_to_i = .01
        self.err_p_i_to_a = .3
        self.prime_timesteps = 0
        self.prime_time = self.prime_timesteps * .1
        self.sim_timesteps = 1
        self.sampling_time = self.sim_timesteps * .1
        self.sampling_period = .1
        self.num_veh_per_lane = 20
        self.max_timesteps = 1000000
        self.hard_brake_threshold = -3.
        self.hard_brake_n_past_frames = 2
        self.ttc_threshold = 3.

        ### heuristic
        self.roadway_radius = 400.
        self.roadway_length = 100.
        self.min_base_speed = 30.
        self.max_base_speed = 30.
        self.min_vehicle_length = 5.
        self.max_vehicle_length = 5.
        self.min_vehicle_width = 2.5
        self.max_vehicle_width = 2.5
        self.min_init_dist = 10.
        self.behavior_type = "heuristic"
        self.heuristic_behavior_type = "correlated"

        ## feature extraction
        self.extractor_type = 'multi'
        self.extract_core = True
        self.extract_temporal = True
        self.extract_well_behaved = True
        self.extract_neighbor = False
        self.extract_behavioral = False
        self.extract_neighbor_behavioral = False
        self.extract_car_lidar = True
        self.extract_car_lidar_range_rate = True
        self.extract_road_lidar = False

        # prediction
        self.hidden_layer_sizes = [128, 128]
        self.value_dim = 5
        self.local_steps_per_update = 20
        self.grad_clip_norm = 40
        self.learning_rate = 1e-4
        self.learning_rate_end = 1e-5
        self.dropout_keep_prob = .95
        self.discount = 49. / 50
        self.n_global_steps = 100000000
        self.summary_every = 11
        self.target_loss_index = 3
        self.l2_reg = 1e-5
        self.eps = 1e-8
        self.loss_type = 'mse'
        self.normalization_type = 'range'

        ## optimizers
        self.optimizer = 'adam'
        self.adam_beta1 = .99
        self.adam_beta2 = .999
        self.adam_epsilon = 1e-8
        self.rmsprop_decay = .9
        self.rmsprop_momentum = 0.

        # monitoring
        self.viz_dir = "videos/"
        self.summarize_features = True
        self.visualize = True

        # validation
        self.validation_dataset_filepath = '/Users/wulfebw/Dropbox/School/Stanford/research/risk/risk_prediction/data/datasets/june/prop_heuristic_5_sec_1_lane.h5'
        self.validate_every = 100000
        self.visualize_every = 1000
        self.max_validation_samples = 100

        # testing
        self.testing = False
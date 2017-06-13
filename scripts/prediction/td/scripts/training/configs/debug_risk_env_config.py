import math

class Config(object):
    def __init__(self):

        # environment options
        self.env_id = 'HeuristicRiskEnv-v0'

        ## generation
        self.num_lanes = 1
        self.max_num_vehicles = 1
        self.min_num_vehicles = 1
        self.base_bn_filepath = "../../data/bayesnets/base_test.jld"
        self.prop_bn_filepath = "../../data/bayesnets/prop_test.jld"
        self.lon_accel_std_dev = 1.
        self.lat_accel_std_dev = 0.
        self.overall_response_time = .0
        self.lon_response_time = .0
        self.err_p_a_to_i = .0
        self.err_p_i_to_a = .0
        # prime_timesteps are the number of initial timesteps used for burn in
        self.prime_timesteps = 300
        # sim_timesteps is the number of timesteps that are simulated 
        # for each call to step
        self.sim_timesteps = 1
        self.num_veh_per_lane = 1
        self.max_timesteps = int(math.ceil(10000 / self.sim_timesteps))
        self.hard_brake_threshold = 0. # -3.09
        self.hard_brake_n_past_frames = 1
        self.ttc_threshold = 3.

        ### heuristic
        self.roadway_radius = 400.
        self.roadway_length = 100.
        self.min_num_veh = 1
        self.max_num_veh = 1
        self.min_base_speed = 30.
        self.max_base_speed = 30.
        self.min_vehicle_length = 5.
        self.max_vehicle_length = 5.
        self.min_vehicle_width = 2.5
        self.max_vehicle_width = 2.5
        self.min_init_dist = 10.
        self.heuristic_behavior_type = "normal"

        ## evaluator
        self.n_monte_carlo_runs = 10

        ## feature extraction
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
        self.hidden_layer_sizes = [128, 64]
        self.value_dim = 5
        self.local_steps_per_update = 1
        self.grad_clip_norm = 40
        self.learning_rate = 1e-3 # / self.local_steps_per_update
        self.dropout_keep_prob = .5
        self.l2_reg = 0.
        self.target_loss_index = 3
        self.eps = 1e-8
        self.loss_type = 'mse'
        # TODO: discount factor seems that it should depend upon sim_timesteps 
        # as well 
        self.discount = 0.
        self.n_global_steps = 100000000
        self.summary_every = 11
        self.normalization_type = 'range'

        ## optimizers
        self.optimizer = 'adam'
        self.adam_beta1 = .995
        self.adam_beta2 = .999
        self.adam_epsilon = 1e-8
        self.rmsprop_decay = .9
        self.rmsprop_momentum = 0.99

        # monitoring
        self.viz_dir = "videos/"
        self.summarize_features = False

        # testing
        self.testing = False
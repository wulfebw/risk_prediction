

class Config(object):
    def __init__(self):
        # environment options
        ## generation
        self.num_lanes = 1
        self.max_num_vehicles = 20
        self.base_bn_filepath = "../data/bayesnets/base_test.jld"
        self.prop_bn_filepath = "../data/bayesnets/prop_test.jld"
        self.lon_accel_std_dev = 1.
        self.lat_accel_std_dev = .1
        self.overall_response_time = .2
        self.lon_response_time = .2
        self.err_p_a_to_i = .01
        self.err_p_i_to_a = .3
        self.prime_timesteps = 0
        self.sim_timesteps = 5
        self.num_veh_per_lane = 10
        self.max_timesteps = 50

        ## feature extraction
        self.extract_core = True
        self.extract_temporal = True
        self.extract_well_behaved = True
        self.extract_neighbor = True
        self.extract_behavioral = True
        self.extract_neighbor_behavioral = True
        self.extract_car_lidar = True
        self.extract_car_lidar_range_rate = True
        self.extract_road_lidar = False

        ## monitoring
        self.viz_dir = "../data/viz/test/"
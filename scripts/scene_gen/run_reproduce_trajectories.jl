include("reproduce_trajectories.jl")
dataset_filepath = "../../data/datasets/march/risk_5_sec_3_timesteps_traj.h5"
output_directory = "../../data/trajdatas/"
seeds_veh_indices_filepath = "../../data/trajdatas/seeds_veh_idxs.csv"
target_index = 3
threshold = .19
targets, seeds, batch_idxs = load_targets_seeds_batch_idxs(dataset_filepath)
seeds, veh_idxs, probs = select_seeds_veh_indices(targets, seeds, batch_idxs, 
    target_index, threshold)
write_seeds_veh_indices_file(seeds_veh_indices_filepath, seeds, veh_idxs, probs)
reproduce_dataset_trajectories(dataset_filepath, output_directory, 
    unique(seeds))
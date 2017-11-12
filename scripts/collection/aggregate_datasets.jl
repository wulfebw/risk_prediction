using AutoRisk

# collect full datasets
basedir = "../../data/datasets/nov/"
input_filepaths = [
    "ngsim_20_sec_1_feature_timesteps_traj_1.h5",
    "ngsim_20_sec_1_feature_timesteps_traj_3.h5",
    "ngsim_20_sec_1_feature_timesteps_traj_2.h5",
    ]
input_filepaths = [string(basedir, f) for f in input_filepaths]
output_filepath = "../../data/datasets/nov/ngsim_20_sec_1_feature_timesteps_traj_1_3_2.h5"

valid_filepaths = Array{String}(0)
for filepath in input_filepaths
    try
        h5open(filepath, "r") do file
        end
        push!(valid_filepaths, filepath)
    catch e
    	println("invalid filepath: $(filepath)")
        println("exception: $(e)")
    end
end

if length(valid_filepaths) > 0
    aggregate_datasets(valid_filepaths, output_filepath)
else
    println("no valid filepaths")
end
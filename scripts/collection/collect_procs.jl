using AutoRisk

# collect full datasets
basedir = "../../data/datasets/may/"
input_filepaths = [
    "ngsim_20_sec_traj_1.h5",
    "ngsim_20_sec_traj_2.h5",
    "ngsim_20_sec_traj_3.h5",
    "ngsim_20_sec_traj_4.h5",
    "ngsim_20_sec_traj_5.h5",
    "ngsim_20_sec_traj_6.h5",
    ]
input_filepaths = [string(basedir, f) for f in input_filepaths]
output_filepath = "../../data/datasets/may/ngsim_20_sec.h5"

# # collect procs
# basedir = "../../data/datasets/"
# num_procs = 18
# input_filenames = ["proc_$(i)_risk_5_second.h5" for i in 1:num_procs]
# input_filepaths = [string(basedir, f) for f in input_filenames]
# output_filepath = "../../data/datasets/risk_5_second_safe.h5"

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
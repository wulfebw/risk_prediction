using AutomotiveDrivingModels
using AutoRisk
using NGSIM

# extraction settings and constants
models = Dict{Int, DriverModel}() # dummy, no behavior available
prime = 25 # /10 = seconds to prime to make all features available
feature_timesteps = 20 # number of timesteps to record features
frameskip = 300 # /10 = seconds to skip between samples
framecollect = 300 # /10 = seconds to collect
frameoffset = 400 # from ends of the trajectories
@assert frameskip >= framecollect
@assert prime >= feature_timesteps + 2
@assert frameoffset >= framecollect

output_filename = "ngsim_$(Int(ceil(framecollect / 10)))_sec_$(feature_timesteps)_feature_timesteps.h5"
output_filepath = joinpath("../../data/datasets/", output_filename)

println("Extracting NGSIM dataset with the following settings:")
println("prime steps: $(prime)")
println("feature steps: $(feature_timesteps)")
println("sampling steps: $(framecollect)")
println("output filepath: $(output_filepath)")

# feature extractor (note the lack of behavioral features)
subexts = [
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        WellBehavedFeatureExtractor(),
        NeighborFeatureExtractor(),
        CarLidarFeatureExtractor()
    ]
ext = MultiFeatureExtractor(subexts)
target_ext = TargetExtractor()

# set the dataset names for the individual trajectories
dataset_filepaths = String[]
for trajdata_index in 1:6
    dataset_filepath = replace(output_filepath, ".h5", "_traj_$(trajdata_index).h5")
    push!(dataset_filepaths, dataset_filepath)
end

tic()
# extract 
@parallel (+) for trajdata_index in 1:6
    # dataset for storing feature => target pairs
    dataset = Dataset(
            dataset_filepaths[trajdata_index],
            length(ext),
            feature_timesteps,
            length(target_ext),
            500000,
            init_file = false,
            attrs = Dict("feature_names"=>feature_names(ext), 
                "target_names"=>feature_names(target_ext),
                "framecollect"=>framecollect))

    trajdata = load_trajdata(trajdata_index)
    roadway = get_corresponding_roadway(trajdata_index)
    max_n_objects = maximum(n_objects_in_frame(trajdata, i) for i in 1 : nframes(trajdata))
    scene = Scene(max_n_objects)
    rec = SceneRecord(prime + framecollect, 0.1, max_n_objects)
    features = zeros(length(ext), feature_timesteps, max_n_objects)
    targets = zeros(5, max_n_objects)
    veh_id_to_idx = Dict{Int,Int}()
    final_frame = nframes(trajdata) - frameoffset
    for initial_frame in frameoffset : frameskip : final_frame
        println("traj: $(trajdata_index) / 6\tframe $(initial_frame) / $(final_frame)")
        # reset
        empty!(veh_id_to_idx)
            
        # prime
        get!(scene, trajdata, initial_frame - prime)
        for frame in (initial_frame - prime):initial_frame
            AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))
        end

        # collect a mapping of vehicle ids to indices in the scene
        get_veh_id_to_idx(scene, veh_id_to_idx)
            
        # extract features
        pull_features!(ext, rec, roadway, models, features, feature_timesteps)
            
        # update with next framecollect frames
        for frame in (initial_frame + 1):(initial_frame + framecollect)
            AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))
        end
            
        # extract targets
        extract_targets!(target_ext, rec, roadway, targets, veh_id_to_idx, true)
            
        # update dataset with features, targets
        actual_num_veh = length(veh_id_to_idx)
        saveframe = trajdata_index * 100000 + initial_frame
        update!(dataset, features[:, :, 1:actual_num_veh], 
            targets[:, 1:actual_num_veh], saveframe)
    end
    finalize!(dataset)
    0 # for @parallel purposes
end

# collect datasets into one
aggregate_datasets(dataset_filepaths, output_filepath)
toc()

using AutomotiveDrivingModels
using AutoRisk
using HDF5
using NGSIM

@everywhere include("dataset_feature_extractors.jl")

# extraction settings and constants
models = Dict{Int, DriverModel}() # dummy, no behavior available

feature_timesteps = 1 # number of timesteps to record features
feature_step_size = 1 # number of timesteps between features
prime = feature_timesteps * feature_step_size + 5 # /10 = seconds to prime to make all features available
framecollect = 200 # /10 = seconds to collect target values
frameskip = framecollect + prime # /10 = seconds to skip between samples
frameoffset = 1000 # from ends of the trajectories
@assert frameoffset >= framecollect

# use previously extracted behavioral features if this filepath != ""
ngsim_behavior_filepath = "../../data/datasets/oct/ngsim_idm_params.h5"

output_filename = "ngsim_$(Int(ceil(framecollect / 10)))_sec_$(feature_timesteps)_feature_timesteps.h5"
output_filepath = joinpath("../../data/datasets/", output_filename)

println("Extracting NGSIM dataset with the following settings:")
println("prime steps: $(prime)")
println("feature steps: $(feature_timesteps)")
println("sampling steps: $(framecollect)")
println("output filepath: $(output_filepath)")

n_trajs = 6

# set the dataset names for the individual trajectories
dataset_filepaths = String[]
for trajdata_index in 1:n_trajs
    dataset_filepath = replace(output_filepath, ".h5", "_traj_$(trajdata_index).h5")
    push!(dataset_filepaths, dataset_filepath)
end

tic()
# extract 
@parallel (+) for trajdata_index in 1:n_trajs

    # subextractors
    subexts = [
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        WellBehavedFeatureExtractor(),
        NeighborFeatureExtractor(),
    ]

    # load behavioral features if available and create extractor for them
    
    if ngsim_behavior_filepath != ""
        ngsim_behavior_features = Dict{Int, Array{Float64}}()
        f = h5open(ngsim_behavior_filepath, "r")
        thetas = f["thetas"]
        ids = f["ids"]
        n_samples = size(ids, 2)
        for i in 1:n_samples
            traj_id, veh_id = tuple(ids[:,i]...)
            if traj_id == trajdata_index
                ngsim_behavior_features[veh_id] = thetas[:,i]
            end
        end
        ngsim_behavior_ext = NGSIMNeighborFeatureExtractor(
            ngsim_behavior_features
        )
        push!(subexts, ngsim_behavior_ext)
    end

    ext = MultiFeatureExtractor(subexts)
    target_ext = TargetExtractor()

    # dataset for storing feature => target pairs
    dataset = Dataset(
            dataset_filepaths[trajdata_index],
            length(ext),
            feature_timesteps,
            length(target_ext),
            framecollect,
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
    targets = zeros(length(target_ext), framecollect, max_n_objects)
    final_frame = nframes(trajdata) - frameoffset
    for initial_frame in frameoffset : frameskip : final_frame
        println("traj: $(trajdata_index) / $(n_trajs)\tframe $(initial_frame) / $(final_frame)")

        # reset scene record and scene
        empty!(rec)
        empty!(scene)
            
        # get the relevant scene
        get!(scene, trajdata, initial_frame - prime)

        # collect a mapping of vehicle ids to indices in the scene prior to 
        # priming, and then intersect this with the set that are present 
        # after priming
        veh_id_to_idx_before = get_veh_id_to_idx(scene, Dict{Int,Int}())

        # prime
        for frame in (initial_frame - prime):(initial_frame - 1)
            AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))
        end

        # collect a mapping of vehicle ids to indices in the scene after 
        # priming
        veh_id_to_idx_after = get_veh_id_to_idx(scene, Dict{Int,Int}())

        # only want vehicles in the scene both before and after priming 
        # this means their id must be present in both
        # as for the index, take that of the after dictionary 
        veh_id_to_idx = Dict{Int,Int}()
        for (id, idx) in veh_id_to_idx_after
            if in(id, keys(veh_id_to_idx_before))
                veh_id_to_idx[id] = idx
            end
        end

        # as a result of this, only a subset of the features extracted will be 
        # valid, so to account for this need those valid idxs for use in subselect
        valid_idxs = collect(values(veh_id_to_idx))
            
        # extract features
        pull_features!(ext, rec, roadway, models, features, feature_timesteps, step_size=feature_step_size)
            
        # update with next framecollect frames
        for frame in initial_frame:(initial_frame + framecollect - 1)
            AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))
        end
            
        # extract targets
        start_frame = rec.nframes - (prime + 1)
        extract_targets!(target_ext, rec, roadway, targets, veh_id_to_idx, true, start_frame)
            
        # update dataset with features, targets
        saveframe = trajdata_index * 100000 + initial_frame
        # actual_num_veh = length(veh_id_to_idx)
        # update!(dataset, features[:, :, 1:actual_num_veh], 
        #     targets[:, 1:actual_num_veh], saveframe)
        update!(dataset, features[:, :, valid_idxs], 
            targets[:, :, valid_idxs], saveframe)
    end
    finalize!(dataset)
    0 # for @parallel purposes
end

# collect datasets into one
aggregate_datasets(dataset_filepaths, output_filepath)
toc()

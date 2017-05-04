using AutomotiveDrivingModels
using AutoRisk
using NGSIM


# extraction settings and constants
models = Dict{Int, DriverModel}() # dummy, no behavior available
prime = 5 # .5 second prime to compute all features
frameskip = 50 # 5 second skip
frameoffset = 400

# feature extractor (note the lack of behavioral features)
subexts = [
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        WellBehavedFeatureExtractor(),
        NeighborFeatureExtractor(),
        CarLidarFeatureExtractor(),
        RoadLidarFeatureExtractor()
    ]
ext = MultiFeatureExtractor(subexts)

# extract 
dataset_filepaths = String[]
tic()
for trajdata_index in 1 : 6
    # dataset for storing feature => target pairs
    dataset_filepath = "../../data/datasets/may/ngsim_$(Int(ceil(frameskip / 10)))_sec_traj_$(trajdata_index).h5"
    push!(dataset_filepaths, dataset_filepath)
    dataset = Dataset(
            dataset_filepath,
            length(ext),
            1,
            5,
            100000,
            init_file = false,
            attrs = Dict("feature_names"=>feature_names(ext)))

    trajdata = load_trajdata(trajdata_index)
    roadway = get_corresponding_roadway(trajdata_index)
    max_n_objects = maximum(n_objects_in_frame(trajdata, i) for i in 1 : nframes(trajdata))
    scene = Scene(max_n_objects)
    rec = SceneRecord(prime + frameskip, 0.1, max_n_objects)
    features = zeros(length(ext), 1, max_n_objects)
    targets = zeros(5, max_n_objects)
    veh_id_to_idx = Dict{Int,Int}()
    final_frame = nframes(trajdata) - frameoffset
    for initial_frame in frameoffset : frameskip : final_frame
        println("traj: $(trajdata_index) / 6\tframe $(initial_frame) / $(final_frame)")
        # reset
        empty!(veh_id_to_idx)
            
        # collect a mapping of vehicle ids to indices in the scene
        get!(scene, trajdata, initial_frame - prime)
        get_veh_id_to_idx(scene, veh_id_to_idx)
        
        # prime
        for frame in (initial_frame - prime):initial_frame
            AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))
        end
            
        # extract features
        pull_features!(ext, rec, roadway, models, features, 1)
            
        # update with next frameskip frames
        for frame in initial_frame:initial_frame + frameskip
            AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))
        end
            
        # extract targets
        extract_targets!(rec, roadway, targets, veh_id_to_idx, true)
            
        # update dataset with features, targets
        actual_num_veh = length(veh_id_to_idx)
        update!(dataset, features[:, :, 1:actual_num_veh], 
            targets[:, 1:actual_num_veh], 0)
    end
    finalize!(dataset)
end
toc()

# aggregate accross datasets
# aggregate_datasets()
#=
For these scripts to work the following should hold:
1. the machine where the data was collected and this machine must give identical simulation results. Reasons why this might not be the case include
- different AutoRisk versions
- different ADM versions
- different versions of any other library
=#
using AutoRisk

include("../collection/collect_heuristic_dataset.jl")

function write_seeds_veh_indices_file(output_filepath::String, 
        seeds::Array{Int}, veh_idxs::Array{Int})
    outfile = open(output_filepath, "w")
    write(outfile, "seed,vehicle_index\n")
    for (s, v) in zip(seeds, veh_idxs)
        write(outfile, "$(s),$(v)\n")
    end
end

function load_targets_seeds_batch_idxs(filepath::String)
    targets, seeds, batch_idxs = [], [], []
    h5open(filepath, "r") do infile
        targets = read(infile, "risk/targets")
        seeds = read(infile, "risk/seeds")
        batch_idxs = read(infile, "risk/batch_idxs")
    end
    return targets, seeds, batch_idxs
end

"""
Description:
    - Selects seeds and corresponding vehicle indices from a dataset based 
        on whether the corresponding value of a target is above a threshold
"""
function select_seeds_veh_indices(targets::Array{Float64}, seeds::Array{Int}, 
        batch_idxs::Array{Int}, target_index::Int = 1, threshold::Float64 = 1.)
    target_seeds = Int[]
    veh_idxs = Int[]
    num_targets, num_samples = size(targets)
    cur_idx = 1
    for sidx in 1:num_samples
        # if the sample is over this batch index then increment the 
        # seed / batch counter e.g., 201 > 200 -> cur_idx should be 2
        if sidx > batch_idxs[cur_idx]
            cur_idx += 1
        end

        # if the target is over the threshold, then collect the seed and 
        # vehicle index
        if targets[target_index, sidx] >= threshold
            # edge case when collecting vehicle index in first seed
            if cur_idx == 1
                veh_idx = sidx
            else
                # e.g., cur_idx = 2, sidx = 201, then veh_idx = 201 - 200 = 1
                veh_idx = sidx - batch_idxs[cur_idx - 1]
            end

            push!(target_seeds, seeds[cur_idx])
            push!(veh_idxs, veh_idx)
        end
    end
    return target_seeds, veh_idxs
end

function simulate!(scene::Scene, models::Dict{Int, DriverModel}, 
        roadway::Roadway, trajdata::Trajdata, timesteps::Int)

    for veh in scene
        trajdata.vehdefs[veh.def.id] = veh.def
    end

    actions = Array(DriveAction, length(scene))
    frame_index, state_index = 0, 0
    for t in 1:timesteps
        lo = state_index + 1
            for veh in scene
                trajdata.states[state_index+=1] = TrajdataState(veh.def.id, veh.state)
            end
        hi = state_index
        trajdata.frames[frame_index+=1] = TrajdataFrame(lo, hi, t)

        get_actions!(actions, scene, roadway, models)
        tick!(scene, roadway, actions, models)
    end

    return trajdata
end

function write_trajdata(trajdata::Trajdata, output_filepath::String)
    outfile = open(output_filepath, "w")
    write(outfile, trajdata)
    close(outfile)
end    

function reproduce_dataset_trajectories(dataset_filepath::String, 
        output_directory::String, seeds::Array{Int})
    # load the dataset and it's attributes, and convert to flags
    flags = h5readattr(dataset_filepath, "risk")
    fixup_types!(flags)

    # build an identical collector
    col = build_dataset_collector("/tmp/reproduction.h5", flags)

    # for each seed in the set of seeds, regenerate the scene, roadway, models
    # and then simulate the scenario for prime_time + sampling_time seconds 
    # collecting the last sampling_time seconds worth of frames in a trajdata
    # then save the trajdata to file in addition to saving the relevant vehicle 
    # indices to file as well
    timesteps = length(col.eval.rec.scenes)
    for seed in seeds
        rand!(col, seed)
        srand(seed)
        trajdata = Trajdata(col.roadway, Dict{Int, VehicleDef}(),
            Array(TrajdataState, length(col.scene) * timesteps),
            Array(TrajdataFrame, timesteps))
        simulate!(col.scene, col.models, col.roadway, trajdata, timesteps)
        output_filepath = joinpath(output_directory, "$(seed).txt")
        write_trajdata(trajdata, output_filepath)
    end
end

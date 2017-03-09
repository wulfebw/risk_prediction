
using AutoRisk

include("../collection/heuristic_dataset_config.jl")
include("../collection/collect_heuristic_dataset.jl")

# read in the seeds and veh indices
tidx = 0
input_filepath = "../../data/visualizations/risk_5_sec_3_timesteps/seed_veh_idx_target_$(tidx).csv"
# input_filepath = "../../data/visualizations/risk/seed_veh_idx_target_$(tidx).csv"
df = readtable(input_filepath)

# should really use config files
# buil the collector
parse_flags!(FLAGS, [])
FLAGS["num_monte_carlo_runs"] = 1
FLAGS["prime_time"] = 10.
FLAGS["sampling_time"] = 5.
FLAGS["overall_response_time"] = 0.2
FLAGS["lon_response_time"] = 0.2
FLAGS["err_p_a_to_i"] =  0.01
FLAGS["err_p_i_to_a"] =  0.3
FLAGS["behavior_type"] = "heuristic"
FLAGS["extractor_type"] =  "multi"
col = build_dataset_collector("", FLAGS);

function upload_rec!(trajdata::Trajdata, rec::SceneRecord, timesteps::Int)
    for veh in rec.scenes[1]
        trajdata.vehdefs[veh.def.id] = veh.def
    end

    state_idx, frame_idx = 0, 0
    for t in 1:timesteps
        lo = state_idx + 1
        for veh in get_scene(rec, t - timesteps)
            trajdata.states[state_idx+=1] = TrajdataState(veh.def.id, veh.state)
        end
        hi = state_idx
        trajdata.frames[frame_idx+=1] = TrajdataFrame(lo, hi, t / 10.)
    end
    return trajdata
end

# for each seed-veh_idx pair
# simulate for prime_time
# extract the scene from the rec
timesteps = Int(ceil(10 * (FLAGS["prime_time"] + FLAGS["sampling_time"])))
output_directory = "../../data/trajdatas"
veh_idx_filepath = string(output_directory, "/veh_idxs.csv")
vehfile = open(veh_idx_filepath, "w")
write(vehfile, "veh_idx\n")
for (i, seed) in enumerate(df[:seed])
    output_filepath = string(output_directory, "/traj_$(i).txt")
    outfile = open(output_filepath, "w")
    println(output_filepath)
    rand!(col, seed)
    evaluate!(col.eval, col.scene, col.models, col.roadway, seed)
    trajdata = Trajdata(col.roadway, Dict{Int, VehicleDef}(),
        Array(TrajdataState, length(col.scene) * timesteps),
        Array(TrajdataFrame, timesteps))
    trajdata = upload_rec!(trajdata, col.eval.rec, timesteps)
    write(outfile, trajdata)
    close(outfile)
    write(vehfile, "$(df[:veh_index][i])\n")
    flush(vehfile)
end
close(vehfile)

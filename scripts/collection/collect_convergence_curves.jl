"""
This is a script to collect convergence curves.
"""

using PGFPlots
using TikzPictures

using AutoRisk
using CommandLineFlags

include("collect_dataset.jl")
include("dataset_config.jl")

const TARGET_NAMES = [
    "Lane Change Collision",
    "Rear End Ego Vehicle in Front",
    "Rear End Ego Vehicle in Rear",
    "Hard Brake",
    "Low Time to Collision"
]

function collect_targets(col, seed)
    # generate scene and unpack
    rand!(col, seed)
    eval, scene, models, roadway = (col.eval, col.scene, col.models, 
        col.roadway)

    # reset values
    reset!(eval, scene, seed)
    
    # prime the scene by simulating for short period
    # extract prediction features at this point
    simulate!(Any, eval.rec, scene, roadway, models, eval.prime_time)

    # need this dictionary because cars may enter or exit the 
    # scene. As a result, the indices of the scene may or may 
    # not correspond to the correct vehicle ids at the end of 
    # each monte carlo run. Note that this must be performed 
    # _after_ the burn-in period since vehicles may leave 
    # the scene during that process.
    get_veh_id_to_idx(scene, eval.veh_id_to_idx)

    # targets to return
    targets = zeros(Float64, (eval.num_runs, size(eval.targets, 1), 
        length(scene)))
    
    # repeatedly simulate, starting from the final burn-in scene 
    temp_scene = Scene(length(scene.entities))
    pastframe = 0 # first iteration, don't alter record
    for idx in 1:eval.num_runs
        # reset
        copy!(temp_scene, scene)
        push_forward_records!(eval.rec, -pastframe)

        # simulate starting from the final burn-in scene
        simulate!(Any, eval.rec, temp_scene, roadway, models, 
            eval.sampling_time, update_first_scene = false)

        # pastframe is the number of frames that have been simulated
        pastframe = Int(round(eval.sampling_time / eval.rec.timestep))

        # get the initial extraction frame, this will typically be the first 
        # frame following the prime time, but in the case where no time is 
        # simulated, it should be the most recent frame
        start_extract_frame = max(pastframe - 1, 0)

        # extract target values from every frame in the record for every vehicle
        extract_targets!(eval.target_ext, eval.rec, roadway, eval.targets, 
            eval.veh_id_to_idx, eval.veh_idx_can_change, start_extract_frame)

        if any(isnan(eval.targets))
            println(eval.targets)
            println(idx)
            readline()
        end

        targets[idx, :, :] = eval.targets[:, :]
    end
    return targets
end

function compute_convergence_curves(targets)
    num_runs, target_dim, num_veh = size(targets)
    means = zeros(target_dim, num_veh)
    mean_curve = Array{Float64}(size(targets))
    for idx in 1:num_runs
        temp_means = deepcopy(means)
        diff = targets[idx, :, :] - temp_means
        means += diff / idx
        mean_curve[idx, :, :] = means
    end
    return mean_curve
end

function plot_convergence_curve(curve, dir, name, target_names)
    num_runs, target_dim, num_veh = size(curve) 
    for vidx in 1:num_veh
        println("vehicle $(vidx) / $(num_veh)")
        for tidx in 1:target_dim
            println("target mean: $(curve[end, tidx, vidx])")

            # only plot if nonzero target
            if sum(curve[:, tidx, vidx]) < 1e-8
                continue
            end

            a = Axis(
                    Plots.Linear(
                        collect(1:num_runs), 
                        round(curve[:, tidx, vidx], 5), 
                        markSize=.5
                    ), 
                # ymin=0., 
                # ymax=1.1,
                xlabel = "Monte Carlo Simulations", 
                ylabel = "Pr($(target_names[tidx]))", 
                title = "Pr($(target_names[tidx])) vs Monte Carlo Simulation"
            )
            TikzPictures.save(string(dir, "$(name)_vidx_$(vidx)_tidx_$(tidx).pdf"), a)
            TikzPictures.save(string(dir, "$(name)_vidx_$(vidx)_tidx_$(tidx).tex"), a)
        end
    end
end

function plot_mean_convergence_curve(curve, dir, name, target_names)
    curve = mean(curve, 3)
    num_runs, target_dim = size(curve) 
    g = GroupPlot(1, target_dim,
        groupStyle = "horizontal sep = 1.0cm, vertical sep = 1.5cm")
    for tidx in 1:target_dim
        println("$(target_names[tidx])")
        a = Axis(
                Plots.Linear(
                    collect(1:num_runs), 
                    round(curve[:, tidx], 5), 
                    markSize=.5
                ), 
            # xlabel = "Simulations", 
            # ylabel = "$(TARGET_NAMES[tidx])", 
            # title = "$(TARGET_NAMES[tidx])"
        )
        push!(g, a)
    end
    TikzPictures.save(string(dir, "$(name).pdf"), g)
    TikzPictures.save(string(dir, "$(name).tex"), g)
end

function main()

    dataset_filepath = ""

    if dataset_filepath == ""
        parse_flags!(FLAGS, ARGS)
        FLAGS["num_monte_carlo_runs"] = 1000
        FLAGS["prime_time"] = 30.
        FLAGS["num_lanes"] = 1
        FLAGS["sampling_time"] = 60.
        FLAGS["roadway_length"] = 400.
        FLAGS["roadway_radius"] = 100.
        FLAGS["max_num_vehicles"] = 1
        FLAGS["min_num_vehicles"] = 1
        FLAGS["lon_accel_std_dev"] = 1.
        FLAGS["lat_accel_std_dev"] = .0
        FLAGS["err_p_a_to_i"] = .0
        FLAGS["err_p_i_to_a"] = .0
        FLAGS["overall_response_time"] = .0
        FLAGS["hard_brake_threshold"] = -3.090232306168
        FLAGS["hard_brake_n_past_frames"] = 1
        FLAGS["heuristic_behavior_type"] = "normal"

        flags = FLAGS
    else
        flags = h5readattr(dataset_filepath, "risk")
        fixup_types!(flags)
    end

    
    output_filepath = ""
    col = build_dataset_collector(output_filepath, flags)
    targets = nothing
    n_seeds = 20
    for seed in 1:n_seeds
        if targets == nothing
            targets = collect_targets(col, seed)
        else
            targets .+= collect_targets(col, seed)
        end
    end
    targets ./= n_seeds
    
    mean_curve = compute_convergence_curves(targets)
    output_directory = "../../data/visualizations/convergence_curves/"
    plot_convergence_curve(mean_curve, output_directory, "cc",
        TARGET_NAMES)
    plot_mean_convergence_curve(mean_curve, output_directory, "mean",
        TARGET_NAMES)
end

@time main()
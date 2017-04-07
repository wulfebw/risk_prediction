
using PGFPlots
using TikzPictures

using AutoRisk

include("collect_heuristic_dataset.jl")
include("heuristic_dataset_config.jl")

const TARGET_NAMES = [
    "Lane Change Collision",
    "Rear End Ego Vehicle in Front",
    "Rear End Ego Vehicle in Rear",
    "Hard Brake",
    "Low Time to Collision"
]

function collect_targets(col, seed)
    rand!(col, seed)
    eval, scene, models, roadway = col.eval, col.scene, col.models, col.roadway

    # reset values
    srand(seed)
    srand(eval.rng, seed)
    fill!(eval.agg_targets, 0)
    eval.num_veh = length(scene)
    empty!(eval.veh_id_to_idx)
    
    # prime the scene by simulating for short period
    # extract prediction features at this point
    simulate!(scene, models, roadway, eval.rec, eval.prime_time)

    # need this dictionary because cars may enter or exit the 
    # scene. As a result, the indices of the scene may or may 
    # not correspond to the correct vehicle ids at the end of 
    # each monte carlo run. Note that this must be performed 
    # _after_ the burn-in period since vehicles may leave 
    # the scene during that process.
    get_veh_id_to_idx(scene, eval.veh_id_to_idx)

    # targets to return
    targets = zeros(Float64, (eval.num_runs, size(eval.targets, 1), length(scene)))
    
    # repeatedly simulate, starting from the final burn-in scene 
    temp_scene = Scene(length(scene.vehicles))
    for idx in 1:eval.num_runs
        # reset
        copy!(temp_scene, scene)
        empty!(eval.rec)

        # simulate starting from the final burn-in scene
        simulate!(temp_scene, models, roadway, eval.rec, eval.sampling_time)

        # extract target values from every frame in the record for every vehicle
        extract_targets!(eval.rec, roadway, eval.targets, eval.veh_id_to_idx,
            eval.veh_idx_can_change)

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

function plot_convergence_curve(curve, dir, name, ext = "pdf")
    num_runs, target_dim, num_veh = size(curve) 
    for vidx in 1:num_veh
        println("vehicle $(vidx) / $(num_veh)")
        for tidx in 1:target_dim

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
                ymin=0., 
                ymax=1.1,
                xlabel = "Number of Simulations", 
                ylabel = "Pr($(TARGET_NAMES[tidx]))", 
                title = "$(TARGET_NAMES[tidx])"
            )
            TikzPictures.save(string(dir, "$(name)_$(vidx)_$(tidx).$(ext)"), a)
        end
    end
end

function plot_mean_convergence_curve(curve, dir, name, ext = "pdf")
    curve = mean(curve, 3)
    num_runs, target_dim = size(curve) 
    g = GroupPlot(1, target_dim,
        groupStyle = "horizontal sep = 1.0cm, vertical sep = 1.5cm")
    for tidx in 1:target_dim
        println("$(TARGET_NAMES[tidx])")
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
    TikzPictures.save(string(dir, "$(name).$(ext)"), g)
end

function main()
    parse_flags!(FLAGS, ARGS)
    FLAGS["num_monte_carlo_runs"] = 50
    FLAGS["prime_time"] = 5.
    FLAGS["sampling_time"] = 30.
    FLAGS["roadway_length"] = 100.
    FLAGS["roadway_radius"] = 50.
    FLAGS["max_num_vehicles"] = 50
    FLAGS["min_num_vehicles"] = 50
    FLAGS["lon_accel_std_dev"] = 3.
    FLAGS["lat_accel_std_dev"] = 1.
    FLAGS["min_init_dist"] = 7.
    FLAGS["max_vehicle_width"] = 2.9
    FLAGS["max_vehicle_length"] = 7.
    FLAGS["err_p_a_to_i"] = .1

    output_filepath = ""
    col = build_dataset_collector(output_filepath, FLAGS)
    seed = 1
    targets = collect_targets(col, seed)
    mean_curve = compute_convergence_curves(targets)
    output_directory = "../../data/visualizations/convergence_curves/"
    plot_mean_convergence_curve(mean_curve, output_directory, "mean")
    plot_convergence_curve(mean_curve, output_directory, "mean")
end

@time main()
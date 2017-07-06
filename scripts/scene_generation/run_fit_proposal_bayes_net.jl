using CommandLineFlags
using JLD

include("../collection/collect_dataset.jl")
# including the config uses the same flags as are used in collection
# below only defining the additional flags used for CEM 
# when passing arguments, the ones that are passed later in the command 
# are used
include("../collection/dataset_config.jl")
@everywhere include("fit_proposal_bayes_net.jl")

add_entry!(FLAGS, "base_bn_filepath", "../../data/bayesnets/base_test.jld", 
    String, "base bayesnet against which to fit")
add_entry!(FLAGS, "output_filepath", "../../data/bayesnets/prop_test.jld", 
    String, "where to save the prop bayes net")
add_entry!(FLAGS, "viz_dir", "../../data/bayesnets/viz", 
    String, "where to save viz files")
add_entry!(FLAGS, "num_monte_carlo_runs", 1, 
    Int, "num times to run each scene")
add_entry!(FLAGS, "prime_time", 0., 
    Float64, "how long to prime a scene")
add_entry!(FLAGS, "sampling_time", 5.,
    Float64, "how long to simulate a scene and track collisions")
add_entry!(FLAGS, "cem_end_prob", .5,
    Float64, "event probability at which to stop training")
add_entry!(FLAGS, "max_iters", 500,
    Int, "max iterations of cem")
add_entry!(FLAGS, "population_size", 200,
    Int, "pop size of cem")
add_entry!(FLAGS, "top_k_fraction", .5,
    Float64, "fraction of pop to keep (.2 means keep top 20%)")
add_entry!(FLAGS, "n_prior_samples", 80000,
    Int, "number of samples from the base bn to start with")
add_entry!(FLAGS, "n_static_prior_samples", 10000,
    Int, "constant number of prior samples added to dataset")

function fit_proposal_bayes_net(
        base_bn_filepath::Union{String},
        flags::Flags;
        output_filepath::String = "../../data/bayesnets/prop_test.jld",
        viz_dir::String = "../../data/bayesnets/viz",
        num_monte_carlo_runs::Int = 1,
        prime_time::Union{Float64,Void} = nothing,
        sampling_time::Union{Float64,Void} = nothing,
        num_lanes::Union{Int,Void} = nothing,
        y::Float64 = .5,
        max_iters::Int = 500,
        N::Int = 1000,
        top_k_fraction::Float64 = .5,
        target_indices::Vector{Int} = [5],
        n_prior_samples::Int = 60000,
        n_static_prior_samples::Int = 10000
    )
    # # only collect a single timestep
    # flags["feature_timesteps"] = 1
    # flags["feature_step_size"] = 1
    # ensure joint generator
    flags["generator_type"] = "joint"
    # changing prime time may be reasonable because want to generate scenes 
    # at same distribution as stationary distribution of dataset
    if prime_time != nothing
        flags["prime_time"] = prime_time
    end
    # it is possible that the original dataset was generated with different 
    # sampling time than we would want
    if sampling_time != nothing
        flags["sampling_time"] = sampling_time
    end
    # want to start out with the prop bn as the base bn
    if base_bn_filepath == nothing
        flags["prop_bn_filepath"] = flags["base_bn_filepath"]
    else
        flags["prop_bn_filepath"] = base_bn_filepath
        flags["base_bn_filepath"] = base_bn_filepath
    end
    # determines for how many runs each scene should be simulated
    flags["num_monte_carlo_runs"] = num_monte_carlo_runs

    # only extract necessary features
    flags["extract_core"] = true
    flags["extract_temporal"] = false
    flags["extract_behavioral"] = true
    flags["extract_well_behaved"] = false
    flags["extract_neighbor"] = true
    flags["extract_neighbor_behavioral"] = false
    flags["extract_car_lidar"] = false
    flags["extract_road_lidar"] = false

    # debug options mostly
    if num_lanes != nothing
        flags["num_lanes"] = num_lanes
    end

    n_cols = max(1, nprocs() - 1)

    # build feature extractors manually
    exts = AbstractFeatureExtractor[]
    for i in 1:n_cols
        push!(exts, build_extractor(flags))
    end

    # build collectors
    # since we manually extract features, set feature_timesteps to 0, which 
    # results in no feature extraction
    flags["feature_timesteps"] = 0  
    cols = [build_dataset_collector("", flags) for _ in 1:n_cols]

    prop_bn, discs = run_cem(cols, 
        exts,
        y, 
        max_iters = max_iters, 
        N = N, 
        top_k_fraction = top_k_fraction, 
        target_indices = target_indices,
        n_prior_samples = n_prior_samples,
        n_static_prior_samples = n_static_prior_samples,
        output_filepath = output_filepath,
        viz_dir = viz_dir
    )
    col = cols[1]
    JLD.save(output_filepath, "bn", col.gen.prop_bn, "discs", discs)
end

parse_flags!(FLAGS, ARGS)

@time fit_proposal_bayes_net(FLAGS["base_bn_filepath"],
    FLAGS,
    output_filepath = FLAGS["output_filepath"],
    viz_dir = FLAGS["viz_dir"],
    num_monte_carlo_runs = FLAGS["num_monte_carlo_runs"],
    prime_time = FLAGS["prime_time"],
    sampling_time = FLAGS["sampling_time"],
    y = FLAGS["cem_end_prob"],
    N = FLAGS["population_size"],
    max_iters = FLAGS["max_iters"],
    top_k_fraction = FLAGS["top_k_fraction"],
    n_prior_samples = FLAGS["n_prior_samples"],
    n_static_prior_samples = FLAGS["n_static_prior_samples"]
)
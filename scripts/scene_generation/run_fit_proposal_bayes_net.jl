using JLD

@everywhere include("fit_proposal_bayes_net.jl")

function fit_proposal_bayes_net(
        dataset_filepath::String;
        base_bn_filepath::Union{String,Void} = nothing,
        output_filepath::String = "../../data/bayesnets/prop_test.jld",
        num_monte_carlo_runs::Int = 1,
        prime_time::Union{Float64,Void} = nothing,
        sampling_time::Union{Float64,Void} = nothing,
        num_lanes::Union{Int,Void} = nothing,
        y::Float64 = .25,
        max_iters::Int = 100,
        N::Int = 1000,
        top_k_fraction::Float64 = .5,
        target_indices::Vector{Int} = [2,3,4,5],
        n_prior_samples::Int = 10000
    )
    # load flags from an existing dataset
    # these flags should be the ones ultimately used in evaluation
    # and that were used to generate the dataset of the base bn
    flags = h5readattr(dataset_filepath, "risk")
    fixup_types!(flags)
    # only collect a single timestep
    flags["feature_timesteps"] = 1
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
    end
    # determines for how many runs each scene should be simulated
    flags["num_monte_carlo_runs"] = num_monte_carlo_runs
    flags["extract_behavioral"] = true

    # debug options mostly
    if num_lanes != nothing
        flags["num_lanes"] = num_lanes
    end

    n_cols = max(1, nprocs() - 1)
    cols = [build_dataset_collector("", flags) for _ in 1:n_cols]
    prop_bn, discs = run_cem(cols, 
        y, 
        max_iters = max_iters, 
        N = N, 
        top_k_fraction = top_k_fraction, 
        target_indices = target_indices,
        n_prior_samples = n_prior_samples
    )
    col = cols[1]
    JLD.save(output_filepath, "bn", col.gen.prop_bn, "discs", discs)
end

@time fit_proposal_bayes_net(
    "../../data/datasets/risk.h5",
    base_bn_filepath = "../../data/bayesnets/base_test.jld",
    sampling_time = 5.,
    num_lanes = 1
)
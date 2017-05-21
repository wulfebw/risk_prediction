using AutoRisk
using AutoViz
using CommandLineFlags
using Discretizers
using DataStructures
using JLD
using PGFPlots

include("../collection/collect_heuristic_dataset.jl")
include("../collection/heuristic_dataset_config.jl")
@everywhere include("../scene_gen/fit_bayes_net.jl")

const FEATURE_NAMES = ["relvelocity", "forevelocity", "foredistance", "vehlength", 
        "vehwidth", "aggressiveness", "isattentive"]

function visualize_stats(
        stats::DefaultDict{String, Vector{Float64}}, 
        iter::Int
    )
    g = GroupPlot(
            3, 3, groupStyle = "horizontal sep = 1.75cm, vertical sep = 1.5cm")
    for (fidx, feature_name) in enumerate(FEATURE_NAMES)
        a = Axis(Plots.Linear(collect(1:iter), stats[feature_name]), 
            width="8cm",
            height="8cm",
            title="$(feature_name)")
        push!(g, a)
    end
    a = Axis(Plots.Linear(collect(1:iter), stats["utilities"]), 
            width="8cm",
            height="8cm",
            title="utilities")
    push!(g, a)
    a = Axis(Plots.Linear(collect(1:iter), stats["weights"]), 
            width="8cm",
            height="8cm",
            title="weights")
    push!(g, a)

    output_filepath = "../../data/visualizations/bayesnets/iter_$(iter).pdf"
    PGFPlots.save(output_filepath, g)
end

function update_stats(
        stats::DefaultDict{String, Vector{Float64}}, 
        features::AbstractArray{Float64}, 
        utilities::AbstractArray{Float64}, 
        weights::AbstractArray{Float64}
    )
    # update feature means
    for (fidx, feature_name) in enumerate(FEATURE_NAMES)
        push!(stats[feature_name], mean(features[fidx,:]))
    end
    push!(stats["utilities"], mean(utilities))
    push!(stats["weights"], mean(weights))
end

function swap_discretization(bin::Int, 
        src::LinearDiscretizer, 
        dest::LinearDiscretizer
    )
    src_mean = (src.binedges[bin] + src.binedges[bin+1]) / 2.
    return encode(dest, src_mean)
end

function swap_discretization(df::DataFrame, src::Vector{LinearDiscretizer}, 
    dest::Vector{LinearDiscretizer})
    n_samples, n_vars = size(df)
    outdf = DataFrame(df)
    for vidx in 1:n_vars
        for sidx in 1:n_samples
            outdf[sidx, vidx] = swap_discretization(
                df[sidx, vidx], src[vidx], dest[vidx])
        end
    end
    return outdf
end

# selects relevant features and adds to data
@everywhere function extract_bn_features( 
        features::Array{Float64}, 
        feature_names::Array{String},
        index::Int
    )
    bn_features = zeros(7)
    bn_features[1:5] = extract_base_features(features, feature_names)[:,index]
    bn_features[6] = extract_aggressiveness(features, feature_names, 
        rand_aggressiveness_if_unavailable = false)[index]
    bn_features[7] = extract_is_attentive(features, feature_names, 
        rand_attentiveness_if_unavailable = false)[index] 
    return bn_features
end

function run_cem(
        cols::Vector{DatasetCollector}, 
        y::Float64;
        max_iters::Int = 10,
        N::Int = 1000, 
        percentile::Float64 = .50, 
        target_indices::Vector{Int} = [1,2,3,4,5],
        n_prior_samples::Int = 10000
    )
    # initialize
    col = cols[1]
    n_vars = length(keys(col.gen.base_bn.name_to_index))
    n_targets, n_vehicles = size(col.eval.targets)
    top_k = Int(ceil(percentile * N))
    proposal_vehicle_index = get_target_vehicle_index(col.gen, col.roadway)
    prior = rand(col.gen.base_bn, n_prior_samples)
    prev_discs = get_discretizers(col.gen.base_assignment_sampler.var_edges)
    stats = DefaultDict{String, Vector{Float64}}(Vector{Float64})
    
    # allocate containers
    utilities = SharedArray(Float64, N)
    weights = SharedArray(Float64, N)
    data = SharedArray(Float64, n_vars, N)
    for iter in 1:max_iters
        println("\niter: $(iter) / $(max_iters) \ty_hat: $(mean(utilities))")
        
        # reset
        fill!(utilities, 0.)
        fill!(weights, 0.)
        fill!(data, 0.)
        
        # generate and evaluate a scene from the current bayes net
        @parallel (+) for scene_idx in 1:N

            # select the corresponding collector
            col_id = (scene_idx % length(cols)) + 1
            col = cols[col_id]
            
            # compute seed
            seed = (iter - 1) * N + scene_idx
            
            # sample a scene + models
            rand!(col, seed)
            
            # evaluate this scene
            evaluate!(col.eval, col.scene, col.models, col.roadway, seed)
            
            # extract the relevant information
            # utilities[scene_idx] = mean(
            #     col.eval.targets[target_indices, proposal_vehicle_index])
            utilities[scene_idx] = mean(
                col.eval.features[4, proposal_vehicle_index])
            weights[scene_idx] = col.gen.weights[proposal_vehicle_index]
            data[:, scene_idx] = extract_bn_features(
                col.eval.features[:,end,:], 
                feature_names(col.eval.ext), 
                proposal_vehicle_index)
        end

        # update and visualize stats
        update_stats(stats, data, utilities, weights)
        @spawnat 1 visualize_stats(stats, iter)
                
        # select top percentile of population
        # indices = reverse(sortperm(utilities .* weights))[1:top_k]
        indices = reverse(sortperm(utilities))[1:top_k]

        # refit bayesnet
        discs, var_edges = get_discretizers(data[:, indices])
        disc_features = discretize_features(data[:, indices], discs)
        training_data = form_training_data(disc_features)
        println(prev_discs)
        prior = swap_discretization(prior, prev_discs, discs)
        training_data = vcat(prior, training_data)
        # update the prior as running set of prior and training data
        prior = training_data[(top_k + 1):end, :]
        prop_bn = fit_bn(training_data)
        prev_discs = get_discretizers(var_edges)
        for col in cols
            col.gen.prop_bn = prop_bn
            col.gen.prop_assignment_sampler = UniformAssignmentSampler(var_edges)
        end

        # check if the target probability has been sufficiently optimized
        if mean(utilities) > y
            break
        end
    end
    return col.gen.prop_bn
end

function fit_proposal_bayes_net()
    # load flags from an existing dataset
    # these flags should be the ones ultimately used in evaluation
    dataset_filepath = "../../data/datasets/may/bn_aug_5_sec_10_timestep_2.h5"
    flags = h5readattr(dataset_filepath, "risk")
    fixup_types!(flags)
    # only collect a single timestep
    flags["feature_timesteps"] = 1
    # want to start out with the prop bn as the base bn
    flags["prop_bn_filepath"] = flags["base_bn_filepath"]
    # NOTE: this value should be set manually
    # and determines for how many runs each scene should be simulated
    flags["num_monte_carlo_runs"] = 1
    flags["extract_behavioral"] = true

    # debug
    flags["num_lanes"] = 1
    flags["sampling_time"] = .1


    n_cols = max(1, nprocs() - 1)
    cols = [build_dataset_collector("", flags) for _ in 1:n_cols]
    prop_bn = run_cem(cols, 
        10000000., 
        max_iters = 500, 
        N = 200, 
        percentile = .1, 
        target_indices = [5],
        n_prior_samples = 1000
    )
    output_filepath = "../../data/bayesnets/cem_prop_test.jld"
    col = cols[1]
    JLD.save(output_filepath, "bn", col.gen.prop_bn, "var_edges", col.gen.prop_assignment_sampler.var_edges)
end

@time fit_proposal_bayes_net()

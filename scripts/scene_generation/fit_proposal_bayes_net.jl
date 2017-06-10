using AutoRisk
using CommandLineFlags
using Discretizers
# using DataStructures
using DataFrames
using PGFPlots

include("../collection/collect_heuristic_dataset.jl")
include("../collection/heuristic_dataset_config.jl")
include("fit_bayes_net.jl")

const FEATURE_NAMES = [
    "relvelocity", 
    "forevelocity", 
    "foredistance", 
    "vehlength", 
    "vehwidth", 
    "aggressiveness", 
    "isattentive"
]

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

function DataFrame(data::Array{Float64}, var_names::Vector{String})
    df = DataFrame()
    for (index, v) in enumerate(var_names)
        df[Symbol(v)] = data[index, :]
    end
    return df
end

# selects relevant features and adds to data
function extract_bn_features( 
        features::Array{Float64}, 
        feature_names::Array{String},
        index::Int
    )
    bn_features = zeros(7)
    bn_features[1:5] = Array(extract_base_features(features, feature_names)[index,:])
    bn_features[6] = extract_aggressiveness(features, feature_names, 
        rand_aggressiveness_if_unavailable = false)[index]
    bn_features[7] = extract_is_attentive(features, feature_names, 
        rand_attentiveness_if_unavailable = false)[index] 
    return bn_features
end

"""
Description:
    Fit a proposal BN via the cross entropy method. Basically, generate a large
    number of scenes, simulate each of them, the ones that result in the most 
    events (collisions, hard brakes) keep them and throw out the others, then
    fit a new BN to these scenes. Repeat this process until the probability of 
    the event is high enough

Args:
    - cols: vector of dataset collector objects used to simulate and eval scenes
    - y: probability of event at which to stop the process
    - max_iters: how many times to run the process max
    - N: population size
    - top_k_fraction: what fraction of scenes to keep each iteration
        e.g., .25 means keep the top 25% of the samples
    - target_indices: which of the target variables should be considered
        1 = lane change collision, 2 = rear end, etc
    - n_prior_samples: number of samples from the base BN to start out with
    - weight_threshold: we throw out samples where the weight is above this threshold

Retruns:
    - Proposal Bayes net and the discretizers associated with it
"""
function run_cem(
        cols::Vector{DatasetCollector}, 
        y::Float64;
        max_iters::Int = 10,
        N::Int = 1000, 
        top_k_fraction::Float64 = .5, 
        target_indices::Vector{Int} = [1,2,3,4,5],
        n_prior_samples::Int = 10000,
        weight_threshold::Float64 = 10.
    )
    # initialize
    col = cols[1]
    n_vars = length(keys(col.gen.base_bn.name_to_index))
    n_targets, n_vehicles = size(col.eval.targets)
    top_k = Int(ceil(top_k_fraction * N))
    proposal_vehicle_index = get_target_vehicle_index(col.gen, col.roadway)

    # derive prior values
    disc_types = get_disc_types(col.gen.base_assignment_sampler)
    discs = col.gen.base_assignment_sampler.discs
    prior = rand(col.gen.base_bn, n_prior_samples)
    prior = decode(prior, discs)
    disc_types = get_disc_types(col.gen.base_assignment_sampler)
    
    # allocate containers
    stats = DefaultDict{String, Vector{Float64}}(Vector{Float64})
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
            
            # extract utilities and weights
            utilities[scene_idx] = mean(
                col.eval.targets[target_indices, proposal_vehicle_index])
            weights[scene_idx] = col.gen.weights[proposal_vehicle_index]

            data[:, scene_idx] = extract_bn_features(
                col.eval.features[:,end,:], 
                feature_names(col.eval.ext), 
                proposal_vehicle_index)
        end

        # select samples with weight > weight_threshold
        valid_indices = find(weights .< weight_threshold)

        # update and visualize stats
        update_stats(stats, data[:, valid_indices], 
            utilities[valid_indices], weights[valid_indices])
        @spawnat 1 visualize_stats(stats, iter)
                
        # select top fraction of population
        indices = reverse(sortperm(utilities .* weights))
        indices = indices[valid_indices]
        indices = indices[1:top_k]

        # add that set to the prior, and remove older samples
        df_data = DataFrame(data[:, indices], FEATURE_NAMES)
        prior = vcat(prior, df_data)[(top_k + 1):end, :]

        # refit bayesnet and reset in the collectors
        prop_bn, discs = fit_bn(prior, disc_types)
        for col in cols
            col.gen.prop_bn = prop_bn
            col.gen.prop_assignment_sampler = AssignmentSampler(discs)
        end

        # check if the target probability has been sufficiently optimized
        if mean(utilities) > y
            break
        end
    end
    return col.gen.prop_bn, discs
end

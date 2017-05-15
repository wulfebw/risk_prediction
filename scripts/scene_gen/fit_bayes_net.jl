using AutoRisk
using BayesNets
using DataFrames
using Discretizers
using Distributions
using HDF5
using JLD
using PGFPlots

function histogram_data(data::DataFrame, output_filepath::String; 
        debug_size::Int = 100000)
    debug_size = min(debug_size, length(data))
    g = GroupPlot(1, length(data), 
            groupStyle = "horizontal sep = 1.75cm, vertical sep = 1.5cm")
    for name in names(data)
        if name == :isattentive continue end
        vals = convert(Vector{Float64}, data[name][1:debug_size])
        a = Axis(Plots.Histogram(vals), title=string(name))
        push!(g, a)
    end
    PGFPlots.save(output_filepath, g)
end

function load_dataset(input_filepath::String; debug_size::Int = 10000000)
    infile = h5open(input_filepath, "r")
    debug_size = min(debug_size, size(infile["risk/features"], 3))
    features = read(infile["risk/features"])[:,1,1:debug_size]
    targets = read(infile["risk/targets"])[:,1:debug_size]
    return features, targets
end

function load_feature_names(input_filepath::String)
    attributes = h5readattr(input_filepath, "risk")
    feature_names = attributes["feature_names"]
    return feature_names
end

function preprocess_features(
        features::Array{Float64}, 
        targets::Array{Float64}, 
        feature_names::Array{String};
        max_collision_prob::Float64 = 0.,
        min_vel::Float64 = 0.,
        max_vel::Float64 = 30.,
        max_Δvel::Float64 = 3.,
        min_dist::Float64 = 10.,
        max_dist::Float64 = 60.,
        min_len::Float64 = 2.5,
        max_len::Float64 = 6.
    )
    # threshold based on collision probability if applicable
    valid_target_inds = find(sum(targets[1:3,:], 1) / 3. .<= max_collision_prob)

    # threshold velocities
    vel_ind = find(feature_names .== "velocity")[1]
    valid_vel_inds = find(min_vel .< features[vel_ind, :] .< max_vel)

    # threshold relative velocity between leading and trailing vehicles
    fore_m_vel_ind = find(feature_names .== "fore_m_vel")[1]
    valid_rel_vel_inds = find(
        abs(features[vel_ind, :] .- features[fore_m_vel_ind, :]) .< max_Δvel)

    # threshold distances
    dist_ind = find(feature_names .== "fore_m_dist")[1]
    valid_dist_inds = find(min_dist .< features[dist_ind, :] .< max_dist)

    # threshold vehicle lengths
    len_ind = find(feature_names .== "length")[1]
    valid_len_inds = find(min_len .< features[len_ind, :] .< max_len)

    valid_inds = intersect(
        valid_target_inds, 
        valid_vel_inds, 
        valid_rel_vel_inds, 
        valid_dist_inds,
        valid_len_inds)
    features = features[:, valid_inds]
    println("number of samples after thresholding: $(length(valid_inds))")
    return features
end

function extract_base_features(features::Array{Float64}, 
        feature_names::Array{String})
    bn_feature_names = ["velocity", "fore_m_vel", "fore_m_dist", "length", 
        "width"]
    inds = [find(feature_names .== name)[1] for name in bn_feature_names]
    base_data = features[inds,:]
    base_data[1,:] .-= base_data[2,:] # convert velocity to relative velocity
    return base_data
end

function extract_aggressiveness(features::Array{Float64}, 
        feature_names::Array{String};
        rand_aggressiveness_if_unavailable::Bool = true)
    # add aggressivenss by inferring it from politeness
    politeness_index = find(feature_names .== "lane_politeness")
    if !isempty(politeness_index)
        politeness_index = politeness_index[1]
        politness_values = features[politeness_index,:];
        aggressiveness_values = infer_correlated_aggressiveness(politness_values);
    elseif rand_aggressiveness_if_unavailable
        num_samples = size(features, 2)
        aggressiveness_values = rand(num_samples)
    else
        throw(ArgumentError("aggressiveness values not found 
            and random aggressiveness set to false"))
    end

    # reshape for later stacking 
    aggressiveness_values = reshape(aggressiveness_values, 
            (1, length(aggressiveness_values)))

    # # add aggressivenss by inferring it from politeness
    # politeness_index = find(feature_names .== "lane_politeness")[1]
    # politness_values = features[politeness_index,:];
    # aggressiveness_values = infer_correlated_aggressiveness(politness_values);
    # aggressiveness_values = reshape(aggressiveness_values, (1, length(aggressiveness_values)))
   return aggressiveness_values
end

function discretize_features(data::Array{Float64}, num_bins::Array{Int})
    num_variables, num_samples = size(data)
    disc_data = zeros(Int, num_variables, num_samples)
    cutpoints = []
    discs = []
    algo = DiscretizeUniformWidth # DiscretizeUniformCount # 
    for vidx in 1:num_variables
        disc = LinearDiscretizer(binedges(algo(num_bins[vidx]), data[vidx,:]))
        push!(cutpoints, disc.binedges)
        for sidx in 1:num_samples
            c = 0
            val = data[vidx, sidx]
            for (c, (lo, hi)) in enumerate(zip(disc.binedges, disc.binedges[2:end]))
                if lo <= val < hi
                    break
                end
            end
        disc_data[vidx, sidx] = c
        end
    end

    var_edges = Dict{Symbol,Vector{Float64}}()
    var_edges[:relvelocity] = cutpoints[1]
    var_edges[:forevelocity] = cutpoints[2]
    var_edges[:foredistance] = cutpoints[3]
    var_edges[:vehlength] = cutpoints[4]
    var_edges[:vehwidth] = cutpoints[5]
    var_edges[:aggressiveness] = cutpoints[6]

    println("Edges from discretizing: ")
    for (k, v) in var_edges
        println("variable: $(k)")
        println("edges: $(v)\n")
    end
    return disc_data, var_edges
end

# get is_attentive separately since it's discrete
function extract_is_attentive(features::Array{Float64},
        feature_names::Array{String};
        rand_attentiveness_if_unavailable::Bool = true,
        stationary_p_attentive::Float64 = .97)
    num_samples = size(features, 2)
    is_attentive_values = ones(Int, num_samples)
    is_attentive_index = find(feature_names .== "is_attentive")

    if !isempty(is_attentive_index)
        is_attentive_index = is_attentive_index[1]
        for sidx in 1:num_samples
            is_attentive = features[is_attentive_index, sidx] > .5 ? 2 : 1
            is_attentive_values[sidx] = is_attentive
        end
    elseif rand_attentiveness_if_unavailable
        for sidx in 1:num_samples
            is_attentive = rand() < stationary_p_attentive ? 2 : 1
            is_attentive_values[sidx] = is_attentive
        end
    else
        throw(ArgumentError("attentiveness values not found 
            and random attentiveness set to false"))
    end
    return is_attentive_values
end

function form_training_data(disc_data::Array{Int}, 
        is_attentive_values::Array{Int})
    training_data = DataFrame(
        relvelocity = disc_data[1,:], 
        forevelocity = disc_data[2,:],
        foredistance = disc_data[3,:], 
        vehlength = disc_data[4,:],
        vehwidth = disc_data[5,:],
        aggressiveness = disc_data[6,:],
        isattentive = is_attentive_values
    )
    return training_data
end

function fit_bn(training_data::DataFrame)
    bn = fit(DiscreteBayesNet, training_data, 
        (
        # :isattentive=>:foredistance, 
        # :isattentive=>:relvelocity,
        # :aggressiveness=>:foredistance, 
        # :aggressiveness=>:relvelocity,
        :foredistance=>:relvelocity,
        :forevelocity=>:relvelocity,
        :forevelocity=>:foredistance,
        :vehlength=>:vehwidth
        )
    )
    return bn
end

function fit_bn(input_filepath::String, 
        output_filepath::String,
        viz_filepath::String;
        debug_size::Int = 1000000,
        num_bins::Array{Int} = Int[10,10,12,5,5,4],
        rand_aggressiveness_if_unavailable::Bool = true,
        rand_attentiveness_if_unavailable::Bool = true,
        stationary_p_attentive::Float64 = .97
    )
    features, targets = load_dataset(input_filepath, debug_size = debug_size)
    feature_names = load_feature_names(input_filepath)
    features = preprocess_features(features, targets, feature_names)
    base_data = extract_base_features(features, feature_names)
    aggressiveness_values = extract_aggressiveness(features, feature_names,
        rand_aggressiveness_if_unavailable = rand_aggressiveness_if_unavailable)
    data = cat(1, base_data, aggressiveness_values)
    disc_data, var_edges = discretize_features(data, num_bins)
    is_attentive_values = extract_is_attentive(features, feature_names,
        rand_attentiveness_if_unavailable = rand_attentiveness_if_unavailable,
        stationary_p_attentive = stationary_p_attentive)
    training_data = form_training_data(disc_data, is_attentive_values)
    histogram_data(training_data, viz_filepath)
    bn = fit_bn(training_data)
    JLD.save(output_filepath, "bn", bn, "var_edges", var_edges)
end

tic()
fit_bn(
    "../../data/datasets/may/ngsim_1_sec.h5", 
    "../../data/bayesnets/base_test.jld",
    "../../data/bayesnets/feature_histograms.pdf")
toc()





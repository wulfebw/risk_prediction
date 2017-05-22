using AutoRisk
using BayesNets
using DataFrames
using Discretizers
using Distributions
using HDF5
using JLD
using PGFPlots

function histogram_data(
        data::DataFrame, 
        output_filepath::String; 
        debug_size::Int = 100000
    )
    debug_size = min(debug_size, length(data))
    g = GroupPlot(1, length(data), 
            groupStyle = "horizontal sep = 1.75cm, vertical sep = 1.5cm")
    for name in names(data)
        if name == :isattentive 
            continue 
        end
        vals = convert(Vector{Float64}, data[name][1:debug_size])
        a = Axis(Plots.Histogram(vals), title=string(name))
        push!(g, a)
    end
    PGFPlots.save(output_filepath, g)
end

function load_dataset(input_filepath::String; debug_size::Int = 10000000)
    infile = h5open(input_filepath, "r")
    debug_size = min(debug_size, size(infile["risk/features"], 3))
    features = read(infile["risk/features"])[:,end,1:debug_size]
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

function extract_base_features(
        features::Array{Float64}, 
        feature_names::Array{String};
        bn_feature_names = ["velocity", "fore_m_vel", "fore_m_dist", "length", 
        "width"]
    )
    inds = [find(feature_names .== name)[1] for name in bn_feature_names]
    base_data = features[inds,:]
    base_data[1,:] .-= base_data[2,:] # convert velocity to relative velocity
    base_data = DataFrame(
        relvelocity = base_data[1,:],
        forevelocity = base_data[2,:],
        foredistance = base_data[3,:],
        vehlength = base_data[4,:],
        vehwidth = base_data[5,:]
    )
    return base_data
end

function extract_aggressiveness(features::Array{Float64}, 
        feature_names::Array{String};
        rand_aggressiveness_if_unavailable::Bool = true)
    # add aggressivenss by inferring it from politeness
    politeness_index = find(feature_names .== "beh_lane_politeness")
    if !isempty(politeness_index)
        politeness_index = politeness_index[1]
        politness_values = features[politeness_index,:];
        aggressiveness_values = infer_correlated_aggressiveness(politness_values);
    elseif rand_aggressiveness_if_unavailable
        num_samples = size(features, 2)
        aggressiveness_values = clamp(randn(num_samples) + .5, 0, 1)
    else
        throw(ArgumentError("aggressiveness values not found 
            and random aggressiveness set to false"))
    end

   return aggressiveness_values
end

# get is_attentive separately since it's discrete
function extract_is_attentive(features::Array{Float64},
        feature_names::Array{String};
        rand_attentiveness_if_unavailable::Bool = true,
        stationary_p_attentive::Float64 = .97)
    num_samples = size(features, 2)
    is_attentive_values = ones(Int, num_samples)
    is_attentive_index = find(feature_names .== "beh_is_attentive")

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

function Discretizers.encode(
        df::DataFrame, 
        discs::Dict{Symbol, AbstractDiscretizer}
    )
    enc_df = DataFrame()
    for var_name in names(df)
        enc_df[var_name] = encode(discs[var_name], df[var_name])
    end
    return enc_df
end

function Discretizers.decode(
        df::DataFrame, 
        discs::Dict{Symbol,AbstractDiscretizer}
    )
    dec_df = DataFrame()
    for var in names(df)
        dec_df[var] = decode(discs[var], df[var])
    end
    return dec_df
end

function get_discretizers(
        data::DataFrame,
        disc_types::Dict{Symbol,DataType};
        discalg::DiscretizationAlgorithm = DiscretizeBayesianBlocks()
    )
    discs = Dict{Symbol, AbstractDiscretizer}()

    for var in names(data)
        if disc_types[var] == LinearDiscretizer{Float64,Int}
            discs[var] = LinearDiscretizer(binedges(discalg, data[var]))
        elseif disc_types[var] == CategoricalDiscretizer{Int,Int}
            discs[var] = CategoricalDiscretizer(data[var])
        else
            throw(ArgumentError("$(disc_types[var]) not implemented"))
        end
    end
    return discs
end

function get_discretizers(
        data::DataFrame, 
        disc_types::Dict{Symbol,DataType},
        n_bins::Dict{Symbol,Int}
    )

    discs = Dict{Symbol, AbstractDiscretizer}()
    for var in names(data)
        if disc_types[var] == LinearDiscretizer{Float64,Int}
            # maps continuous to discrete bins
            low = minimum(data[var])
            high = maximum(data[var])
            discs[var] = LinearDiscretizer(linspace(low, high, n_bins[var] + 1))
        elseif disc_types[var] == CategoricalDiscretizer{Int,Int}
            # identity mapping between bins
            discs[var] = CategoricalDiscretizer(data[var])
        else
            throw(ArgumentError("$(disc_types[var]) not implemented"))
        end
    end
    return discs
end

function fit_bn(
        data::DataFrame,
        disc_types::Dict{Symbol,DataType};
        n_bins::Dict{Symbol,Int} = Dict{Symbol,Int}(),
        edges = (
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
    # if n_bins not provided, infer binning from data
    if isempty(n_bins)
        discs = get_discretizers(data, disc_types)
    else
        discs = get_discretizers(data, disc_types, n_bins)
    end

    # print out
    for (k, v) in discs
        println("variable: $(k)")
        if isa(v, LinearDiscretizer)
            println("edges: $(v.binedges)\n")
        elseif isa(v, CategoricalDiscretizer)
            println("mapping: $(v.d2n)\n")
        end
    end
    
    # encode and fit a discrete bayes net
    disc_data = encode(data, discs)
    bn = fit(DiscreteBayesNet, disc_data, edges)
    return bn, discs
end

function fit_bn(
        input_filepath::String, 
        output_filepath::String,
        viz_filepath::String;
        debug_size::Int = 500000,
        n_bins::Dict{Symbol,Int} = Dict(
            :relvelocity=>12,
            :forevelocity=>12,
            :foredistance=>12,
            :vehlength=>10,
            :vehwidth=>8,
            :aggressiveness=>5,
            :isattentive=>2
        ),
        disc_types::Dict{Symbol,DataType} = Dict(
            :relvelocity=>LinearDiscretizer{Float64,Int},
            :forevelocity=>LinearDiscretizer{Float64,Int},
            :foredistance=>LinearDiscretizer{Float64,Int},
            :vehlength=>LinearDiscretizer{Float64,Int},
            :vehwidth=>LinearDiscretizer{Float64,Int},
            :aggressiveness=>LinearDiscretizer{Float64,Int},
            :isattentive=>CategoricalDiscretizer{Int,Int}
        ),
        rand_aggressiveness_if_unavailable::Bool = true,
        rand_attentiveness_if_unavailable::Bool = true,
        stationary_p_attentive::Float64 = .97,
        edges = (
            # :isattentive=>:foredistance, 
            # :isattentive=>:relvelocity,
            # :aggressiveness=>:foredistance, 
            # :aggressiveness=>:relvelocity,
            :foredistance=>:relvelocity,
            :forevelocity=>:relvelocity,
            :forevelocity=>:foredistance,
            :vehlength=>:vehwidth,
            :vehlength=>:foredistance
        )
    )
    # load and preprocess
    features, targets = load_dataset(input_filepath, debug_size = debug_size)
    feature_names = load_feature_names(input_filepath)
    features = preprocess_features(features, targets, feature_names)
    
    # formulate data
    base_data = extract_base_features(features, feature_names)
    aggressiveness_values = extract_aggressiveness(features, feature_names,
        rand_aggressiveness_if_unavailable = rand_aggressiveness_if_unavailable)
    base_data[:aggressiveness] = aggressiveness_values
    is_attentive_values = extract_is_attentive(features, feature_names,
        rand_attentiveness_if_unavailable = rand_attentiveness_if_unavailable,
        stationary_p_attentive = stationary_p_attentive)
    base_data[:isattentive] = is_attentive_values

    # discretize and fit
    bn, discs = fit_bn(base_data, disc_types, n_bins = n_bins, edges = edges)
    
    # save
    JLD.save(output_filepath, "bn", bn, "discs", discs)
end
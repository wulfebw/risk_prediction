using AutoRisk
using BayesNets
using DataFrames
using Discretizers
using Distributions
using HDF5
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
    targets = read(infile["risk/targets"])[:,:,1:debug_size]
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
        max_collision_prob::Float64 = 1.,
        min_vel::Float64 = 3.,
        max_vel::Float64 = 40.,
        max_Δvel::Float64 = 6.,
        min_dist::Float64 = 4.,
        max_dist::Float64 = 40.,
        min_len::Float64 = 2.5,
        max_len::Float64 = 6.
    )
    # first check that the necessary features are in the dataset
    msg = "Dataset missing required feature"
    @assert in("velocity", feature_names) msg
    @assert in("fore_m_vel", feature_names) msg
    @assert in("fore_m_dist", feature_names) msg
    @assert in("length", feature_names) msg

    # threshold based on collision probability if applicable
    valid_target_inds = find(sum(targets[1:3,:,:], (1,2)) / 3. .<= max_collision_prob)

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
        aggressiveness_values = clamp(randn(num_samples) * .1 + .5, 0, 1)
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
        discs::Dict{Symbol, LinCatDiscretizer}
    )
    enc_df = DataFrame()
    for var_name in names(df)
        enc_df[var_name] = encode(discs[var_name], df[var_name])
    end
    return enc_df
end

function Discretizers.decode(
        df::DataFrame, 
        discs::Dict{Symbol, LinCatDiscretizer}
    )
    dec_df = DataFrame()
    for var in names(df)
        dec_df[var] = decode(discs[var], df[var])
    end
    return dec_df
end

function get_discretizer(
        data::AbstractArray,
        disc_type::DataType,
        nbin::Union{Void,Int};
        discalg::DiscretizationAlgorithm = DiscretizeBayesianBlocks(),
        max_data_size::Int = 10000
    )
    
    if disc_type == LinearDiscretizer{Float64,Int}
        # if nbin not provided then infer from data
        if nbin == nothing

            # compute bins using only a subset of the data because it the 
            # bayesian blocks algorithm doesn't seem to scale that well
            data_size = length(data)
            data_size = min(data_size, max_data_size)
            disc = LinearDiscretizer(binedges(discalg, data[1:data_size]))
        else
            # maps continuous to discrete bins
            low = minimum(data)
            high = maximum(data)

            # handle the low = high case by creating a single bin containing
            # low to high + epsilon
            # not a great solution, though this should not come up in practice
            # because it means that the variable only takes on a single value 
            # and therefore should likely not be part of the BN
            if low == high
                disc = LinearDiscretizer(linspace(low, high + 1e-8, 2))
                println("Warning: single bin in fitting bayes net")
            else
                disc = LinearDiscretizer(linspace(low, high, nbin + 1))
            end
        end
    elseif disc_type == CategoricalDiscretizer{Int,Int}
        # assumes integer valued
        disc = CategoricalDiscretizer(convert(Array{Int}, data))
    else
        throw(ArgumentError("$(disc_type) not implemented"))
    end    
    return disc
end

function get_discretizers(
        data::DataFrame,
        disc_types::Dict{Symbol,DataType},
        nbins::Dict{Symbol,Int} = Dict{Symbol,Int}()
    )
    discs = Dict{Symbol, LinCatDiscretizer}()
    for var in names(data)
        nbin = in(var, keys(nbins)) ? nbins[var] : nothing
        discs[var] = get_discretizer(data[var], disc_types[var], nbin)
    end
    return discs
end

# function get_discretizers(
#         data::DataFrame,
#         disc_types::Dict{Symbol,DataType};
#         discalg::DiscretizationAlgorithm = DiscretizeBayesianBlocks(),
#         max_data_size::Int = 10000
#     )
#     discs = Dict{Symbol, LinCatDiscretizer}()

#     for var in names(data)
#         if disc_types[var] == LinearDiscretizer{Float64,Int}
#             # compute bins using only a subset of the data because it the 
#             # bayesian blocks algorithm doesn't seem to scale that well
#             data_size = length(data[var])
#             data_size = min(data_size, max_data_size)
#             discs[var] = LinearDiscretizer(
#                 binedges(discalg, data[var][1:data_size]))
#         elseif disc_types[var] == CategoricalDiscretizer{Int,Int}
#             # assumes integer valued
#             discs[var] = CategoricalDiscretizer(convert(Array{Int}, data[var]))
#         else
#             throw(ArgumentError("$(disc_types[var]) not implemented"))
#         end
#     end
#     return discs
# end

# function get_discretizers(
#         data::DataFrame, 
#         disc_types::Dict{Symbol,DataType},
#         n_bins::Dict{Symbol,Int}
#     )

#     discs = Dict{Symbol, LinCatDiscretizer}()
#     for var in names(data)
#         if disc_types[var] == LinearDiscretizer{Float64,Int}

#             # maps continuous to discrete bins
#             low = minimum(data[var])
#             high = maximum(data[var])

#             # handle the low = high case by creating a single bin containing
#             # low to high + epsilon
#             # not a great solution, though this should not come up in practice
#             if low == high
#                 discs[var] = LinearDiscretizer(linspace(low, high + 1e-8, 2))
#                 println("Warning: single bin in fitting bayes net")
#             else
#                 discs[var] = LinearDiscretizer(linspace(low, high, n_bins[var] + 1))
#             end
            
#         elseif disc_types[var] == CategoricalDiscretizer{Int,Int}
#             # identity mapping between bins
#             discs[var] = CategoricalDiscretizer(convert(Array{Int}, data[var]))
#         else
#             throw(ArgumentError("$(disc_types[var]) not implemented"))
#         end
#     end
#     return discs
# end

function fit_bn(
        data::DataFrame,
        discs::Dict{Symbol, LinCatDiscretizer},
        edges = (
            :isattentive=>:foredistance, 
            :aggressiveness=>:foredistance, 
            # :aggressiveness=>:relvelocity,
            # :isattentive=>:relvelocity,
            :foredistance=>:relvelocity,
            :forevelocity=>:relvelocity,
            :forevelocity=>:foredistance,
            :vehlength=>:vehwidth
        )
    )
    disc_data = encode(data, discs)
    bn = fit(DiscreteBayesNet, disc_data, edges)
    return bn
end

function fit_bn(
        data::DataFrame,
        disc_types::Dict{Symbol,DataType};
        n_bins::Dict{Symbol,Int} = Dict{Symbol,Int}(),
        edges = (
            :isattentive=>:foredistance, 
            :aggressiveness=>:foredistance, 
            # :aggressiveness=>:relvelocity,
            # :isattentive=>:relvelocity,
            :foredistance=>:relvelocity,
            :forevelocity=>:relvelocity,
            :forevelocity=>:foredistance,
            :vehlength=>:vehwidth
        )
    )
    # if n_bins is empty this will infer the bins from the data
    discs = get_discretizers(data, disc_types, n_bins)
    
    # encode and fit a discrete bayes net
    bn = fit_bn(data, discs, edges)

    # print out
    for (k, v) in discs
        println("variable: $(k)")
        if isa(v, LinearDiscretizer)
            println("edges: $(v.binedges)\n")
        elseif isa(v, CategoricalDiscretizer)
            println("mapping: $(v.d2n)\n")
        end
    end

    for k in keys(discs)
        println("variable: $(k)")
        println("distributions: \n $(get(bn, k).distributions)")
    end

    return bn, discs
end

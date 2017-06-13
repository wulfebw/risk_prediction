
using JLD

include("fit_bayes_net.jl")

function fit_bn(
        input_filepath::String, 
        output_filepath::String,
        viz_filepath::String;
        debug_size::Int = 1000000,
        n_bins::Dict{Symbol,Int} = Dict(
            :relvelocity=>8,
            :forevelocity=>10,
            :foredistance=>15,
            :vehlength=>8,
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
            :isattentive=>:foredistance, 
            :aggressiveness=>:foredistance, 
            # :isattentive=>:relvelocity,
            # :aggressiveness=>:relvelocity,
            :foredistance=>:relvelocity,
            :forevelocity=>:relvelocity,
            :forevelocity=>:foredistance,
            :vehlength=>:vehwidth,
            :vehlength=>:foredistance
        )
    )
    # load and preprocess
    println("loading data...")
    features, targets = load_dataset(input_filepath, debug_size = debug_size)
    feature_names = load_feature_names(input_filepath)
    println("preprocessing data...")
    features = preprocess_features(features, targets, feature_names)
    
    # formulate data
    println("formating data...")
    base_data = extract_base_features(features, feature_names)
    aggressiveness_values = extract_aggressiveness(features, feature_names,
        rand_aggressiveness_if_unavailable = rand_aggressiveness_if_unavailable)
    base_data[:aggressiveness] = aggressiveness_values
    is_attentive_values = extract_is_attentive(features, feature_names,
        rand_attentiveness_if_unavailable = rand_attentiveness_if_unavailable,
        stationary_p_attentive = stationary_p_attentive)
    base_data[:isattentive] = is_attentive_values

    # discretize and fit
    println("training bayesnet...")
    bn, discs = fit_bn(base_data, disc_types, n_bins = n_bins, edges = edges)
    
    # save
    println("saving...")
    JLD.save(output_filepath, "bn", bn, "discs", discs)
end

@time fit_bn(
    "../../data/datasets/june/heuristic_bn_training.h5", 
    "../../data/bayesnets/heuristic_1_lane.jld",
    "../../data/bayesnets/feature_histograms.pdf"
)

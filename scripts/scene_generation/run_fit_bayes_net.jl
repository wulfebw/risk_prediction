
using CommandLineFlags
using JLD

include("fit_bayes_net.jl")

FLAGS = Flags()
add_entry!(FLAGS, "input_filepath", "../../data/datasets/risk.h5", 
    String, "training data filepath")
add_entry!(FLAGS, "output_filepath", "../../data/bayesnets/base_test.jld", 
    String, "where to save the bayesnet")
add_entry!(FLAGS, "viz_filepath", "../../data/bayesnets/feature_histograms.pdf", 
    String, "where to save visualizations")

function fit_bn(
        input_filepath::String, 
        output_filepath::String,
        viz_filepath::String;
        debug_size::Int = 100000,
        n_bins::Dict{Symbol,Int} = Dict(
            :relvelocity=>16, # 16
            :forevelocity=>14, # 14
            :foredistance=>12, # 12
            :vehlength=>6,
            :vehwidth=>6,
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

parse_flags!(FLAGS, ARGS)

@time fit_bn(
    FLAGS["input_filepath"], 
    FLAGS["output_filepath"],
    FLAGS["viz_filepath"]
)

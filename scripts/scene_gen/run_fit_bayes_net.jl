include("fit_bayes_net.jl")
tic()
fit_bn(
    "../../data/datasets/may/ngsim_1_sec.h5", 
    "../../data/bayesnets/base_test.jld",
    "../../data/bayesnets/feature_histograms.pdf")
toc()

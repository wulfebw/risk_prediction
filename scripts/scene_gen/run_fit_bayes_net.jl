include("fit_bayes_net.jl")
tic()
fit_bn(
    "../../data/datasets/may/heursitic_single_lane_5_sec_10_timestep.h5", 
    "../../data/bayesnets/base_test.jld",
    "../../data/bayesnets/feature_histograms.pdf")
toc()

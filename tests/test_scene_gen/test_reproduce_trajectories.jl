using HDF5
using Base.Test

include("../../scripts/scene_gen/reproduce_trajectories.jl")

function test_select_seeds_veh_indices()
    target_index = 5
    targets = zeros(5, 100)
    targets[target_index, 51] = 1.
    seeds = Int[1,2]
    batch_idxs = Int[50,100]
    seeds, veh_idxs = select_seeds_veh_indices(targets, seeds, batch_idxs,
        target_index)
    @test seeds == Int[2]
    @test veh_idxs == Int[1]
end

@time test_select_seeds_veh_indices()
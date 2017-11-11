# identifier feature extractors
# these extract the id of the ego or neighbor vehicles

using AutomotiveDrivingModels
using AutoRisk

type BehaviorDatasetFeatureExtractor <: AutomotiveDrivingModels.AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function BehaviorDatasetFeatureExtractor()
        num_features = 6
        return new(zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::BehaviorDatasetFeatureExtractor) = ext.num_features
function AutoRisk.feature_names(ext::BehaviorDatasetFeatureExtractor)
    return String[
        "ego_id",
        "velocity",
        "length",
        "fore_m_dist",
        "fore_m_vel",
        "accel"
    ]
end

function AutomotiveDrivingModels.pull_features!(
        ext::BehaviorDatasetFeatureExtractor, 
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0,
        censor_hi_dist::Float64 = 100.)
    scene = rec[pastframe]
    veh_ego = scene[veh_idx]
    idx = 0
    # ego scene values
    ext.features[idx+=1] = veh_ego.id
    ext.features[idx+=1] = veh_ego.state.v
    ext.features[idx+=1] = veh_ego.def.length

    # fore neighbor values
    vtpf = VehicleTargetPointFront()
    vtpr = VehicleTargetPointRear()
    fore_M = get_neighbor_fore_along_lane(scene, veh_idx, roadway, vtpf, vtpr, vtpf)
    if fore_M.ind != 0
        ext.features[idx+=1] = fore_M.Δs
        ext.features[idx+=1] = scene[fore_M.ind].state.v
    else
        # if fore neighbor unavailable, then set the distance to a 
        # censor high value and set the velocity to that of the 
        # ego vehicle so as to have a 0 relative velocity
        ext.features[idx+=1] = censor_hi_dist
        ext.features[idx+=1] = veh_ego.state.v
    end

    ext.features[idx+=1] = convert(Float64, get(
        ACC, rec, roadway, veh_idx, pastframe))

    return ext.features
end


type NGSIMBehaviorFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    behavior::Dict{Int, Array{Float64}}
    mean_beh_features::Array{Float64}
    veh_ids::Set{Int}
    function NGSIMBehaviorFeatureExtractor(behavior::Dict{Int, Array{Float64}})
        num_veh = 11
        num_beh_features = 5
        num_features = num_beh_features * num_veh

        # compute the mean behavior in this trajectory set to use for missing 
        # vehicles (the values should really be zero, but this will likely 
        # work better)
        total = zeros(num_beh_features)
        n = 0
        for (k,v) in behavior
            n += 1
            total += v
        end
        mean_beh = total / n
        return new(
            zeros(Float64, num_features), 
            num_features,
            behavior,
            mean_beh,
            Set(keys(behavior)))
    end
end
Base.length(ext::NGSIMBehaviorFeatureExtractor) = ext.num_features
function AutoRisk.feature_names(ext::NGSIMBehaviorFeatureExtractor)
    neigh_names = ["fore_m", "fore_l", "fore_r", "rear_m", "rear_l", "rear_r",
        "fore_fore_m", "fore_fore_fore_m", "fore_fore_fore_fore_m", "fore_fore_fore_fore_fore_m"]
        beh_feature_names = [
            "beh_lon_a_max", 
            "beh_lon_desired_velocity",
            "beh_lon_s_min",
            "beh_lon_T",
            "beh_lon_d_cmf"
        ]
    fs = String[]
    # ego vehicle behavior
    for subname in beh_feature_names
        push!(fs, "$(subname)")
    end
    # neighbor behavior
    for name in neigh_names
        for subname in beh_feature_names
            push!(fs, "$(name)_$(subname)")
        end
    end

    return fs
end
function AutomotiveDrivingModels.pull_features!(
        ext::NGSIMBehaviorFeatureExtractor,  
        rec::SceneRecord,
        roadway::Roadway, 
        veh_idx::Int,  
        models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(),
        pastframe::Int = 0)

    scene = rec[pastframe]
    ego_id = scene[veh_idx].id
    
    vtpf = VehicleTargetPointFront()
    vtpr = VehicleTargetPointRear()
    fore_M = get_neighbor_fore_along_lane(
        scene, veh_idx, roadway, vtpf, vtpr, vtpf)
    fore_L = get_neighbor_fore_along_left_lane(
        scene, veh_idx, roadway, vtpf, vtpr, vtpf)
    fore_R = get_neighbor_fore_along_right_lane(
        scene, veh_idx, roadway, vtpf, vtpr, vtpf)
    rear_M = get_neighbor_rear_along_lane(
        scene, veh_idx, roadway, vtpr, vtpf, vtpr)
    rear_L = get_neighbor_rear_along_left_lane(
        scene, veh_idx, roadway, vtpr, vtpf, vtpr)
    rear_R = get_neighbor_rear_along_right_lane(
        scene, veh_idx, roadway, vtpr, vtpf, vtpr)

    fore_neigh = fore_M
    fore_neighs = NeighborLongitudinalResult[]
    for i in 1:4
        if fore_neigh.ind != 0
            next_fore_neigh = get_neighbor_fore_along_lane(     
            scene, fore_neigh.ind, roadway, vtpr, vtpf, vtpr)
        else
            next_fore_neigh = NeighborLongitudinalResult(0, 0.)
        end
        push!(fore_neighs, next_fore_neigh)
        fore_neigh = next_fore_neigh
    end

    idxs::Vector{Int64} = [veh_idx, fore_M.ind, fore_L.ind, fore_R.ind, rear_M.ind, 
        rear_L.ind, rear_R.ind]
    idxs = vcat(idxs, [n.ind for n in fore_neighs])

    fidx = 0
    num_neigh_features = 5
    for neigh_veh_idx in idxs
        stop = fidx + num_neigh_features
        if neigh_veh_idx == 0 || !in(scene[neigh_veh_idx].id, ext.veh_ids)
            ext.features[fidx + 1:stop] = ext.mean_beh_features
        else
            veh_id = scene[neigh_veh_idx].id
            beh_features = ext.behavior[veh_id]
            ext.features[fidx + 1:stop] = beh_features
        end
        fidx += num_neigh_features
    end
    return ext.features
end

function normalize_log_probs(w::Array{Float64}, axis::Int = 1)
    exp_values = exp(w .- maximum(w, axis))
    probs = exp_values ./ sum(exp_values, axis)
    return probs
end

rmse(v1::Array{Float64}, v2::Array{Float64}) = mean(sqrt((v1 .- v2)^2))
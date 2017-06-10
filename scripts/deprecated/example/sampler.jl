
# Sampler
abstract Sampler

function Distributions.pdf(sampler::Sampler, x::Array{Float64})
    if length(size(x)) > 1
        D, N = size(x)
        w = ones(N)
        for i in 1:N
            w[i] = prod(pdf(sampler.dists[j], x[j, i]) for j in 1:D)
        end
            return w
    else
        return prod(pdf(sampler.dists[i], x[i]) for i in 1:length(x))
    end
end

function sample!(sampler::Sampler, x::Array{Float64}; xmin::Float64=0., xmax::Float64=0.)
    for (i, dist) in enumerate(sampler.dists)
        x[i] = rand(dist) 
    end
end
# base update does nothing
update!(sampler::Sampler, feedback::Dict{Array{Float64},Float64}) = sampler
update!(sampler::Sampler, feedback::Dict{Array{Float64},Float64}, true_sampler::Sampler) = sampler
reset_dists!(sampler::Sampler, dists::Any) = sampler

# basic sampler
type BasicSampler <: Sampler
    dists::Vector{Distribution}
end

type CEMSampler <: Sampler
    dists::Any
    update_dists_every::Int
    feedback::Dict{Array{Float64},Float64}
    num_components::Int
    cur_idx::Int
    function CEMSampler(dists::Any; update_dists_every::Int = 100, num_components::Int = 2)
        return new(dists, update_dists_every, Dict{Array{Float64},Float64}(), num_components, 0)
    end
end
        
function softmax(values::Array{Float64})
    exp_values = exp(values .- maximum(values))
    probs = exp_values ./ sum(exp_values)
    return probs
end

# update stats associated with the most recent sample
        function update_dists!(sampler::CEMSampler, true_w::Array{Float64}, samples::Array{Float64}, td_errors::Array{Float64})

    utility_weights = softmax(abs(td_errors))
    true_w = true_w .* utility_weights
            
    for (i, dist) in enumerate(sampler.dists)
        proposal_w = pdf(dist, reshape(samples[i,:], 1, length(samples[i,:])))
        samp_w = true_w ./ proposal_w
        samp_w = reshape(samp_w, 1, length(samp_w))
        pis, mus, sigmas = fit_gmm(samples, samp_w = samp_w, num_components = sampler.num_components)
                
        normals = MvNormal[MvNormal(mus[:,k], sigmas[:,:,k]) for k in 1:length(pis)]
        sampler.dists[i] = MixtureModel(normals, reshape(pis, length(pis)))
    end
end

function update!(sampler::CEMSampler, feedback::Dict{Array{Float64},Float64}, true_sampler::Sampler)
    update!(sampler.feedback, feedback)
    if sampler.cur_idx == sampler.update_dists_every
        sampler.cur_idx = 0
        samples = keys(sampler.feedback)
        true_w = pdf(true_sampler, samples)
        num_samples = size(samples, 2)
        td_errors = zeros(Float64, num_samples)
        for (i, s) in enumerate(samples)
            td_errors[i] = feedback[s]
        end
        td_errors = [sampler.feedback[s] for s in samples]
        update_dists!(sampler, true_w, samples, td_errors)
    end
end
function sample!(sampler::CEMSampler, x::Array{Float64}; xmin::Float64=-1., xmax::Float64=1.)
    sampler.cur_idx += 1
    for (i, dist) in enumerate(sampler.dists)
        x[i] = rand(dist)[1]
        x[i] = min(max(env.x[i], env.xmin), xmax)
    end
    return x
end

function reset_dists!(sampler::CEMSampler, dists::Any)
    sampler.dists = copy(dists)
    sampler.cur_idx = 0
end

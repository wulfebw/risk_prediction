# env
abstract Env
@with_kw type OneDEnv <: Env
    xmin::Float64 = -10.
    xmax::Float64 = 10
    x::Array{Float64} = [0.]
    true_sampler::Sampler = BasicSampler([Uniform(xmin, xmax)])
    sampler::Sampler = BasicSampler([Uniform(xmin, xmax)])
end
function reset(env::OneDEnv) 
            sample!(env.sampler, env.x, xmin=env.xmin, xmax=env.xmax)
    env.x
end
function reset(env::OneDEnv, x::Array{Float64})
    env.x = x
end
function update!(env::OneDEnv, feedback::Dict{Array{Float64},Float64})
    update!(env.sampler, feedback, env.true_sampler)
end
function step(env::OneDEnv, a::Array{Float64})
    env.x += a[1]
    if env.x[1] > env.xmax
        r = 1.
        done = true
    elseif env.x[1] < env.xmin
        r = 0.
        done = true
    else
        r = 0.
        done = false
    end
    return (env.x, r, done)
end
abstract Policy
@with_kw type UnivariateGaussianPolicy <: Policy
    μ::Float64 = 0.
    σ::Float64 = 1.
end
step(agent::UnivariateGaussianPolicy, state::Array{Float64}) = [randn() * agent.σ + agent.μ]

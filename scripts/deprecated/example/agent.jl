# Agent
abstract Agent
@with_kw type GaussianOneDAgent <: Agent
    μ::Float64 = 0.
    σ::Float64 = 1.
end
step(agent::GaussianOneDAgent, state::Array{Float64}) = [randn() * agent.σ + agent.μ]

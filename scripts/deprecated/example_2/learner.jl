abstract Learner

type TDLearner <: Learner
    grid::RectangleGrid # for intepolating continuous states
    values::Array{Float64} # for maintaining state values (target dim, num unique states)
    targets::Array{Float64} # temp container for returning target values
    target_dim::Int # dimension of output
    lr::Float64 # learning rate for td update
    discount::Float64 # discount rate (not entirely sound here)
    td_errors::Array{Float64} # td errors
    function TDLearner(grid::RectangleGrid, target_dim::Int;
            lr::Float64 = .1, discount::Float64 = 1.)
        values = zeros(Float64, target_dim, length(grid))
        targets = zeros(Float64, target_dim)
        td_errors = Float64[]
        return new(grid, values, targets, target_dim, lr, discount, td_errors)
    end
end

reset!(learner::TDLearner) = fill!(learner.values, 0)
function get_feedback(learner::TDLearner)
    td_errors = learner.td_errors[:]
    empty!(learner.td_errors)
    return td_errors
end

function predict(learner::TDLearner, state::Vector{Float64})
    for tidx in 1:learner.target_dim
        inds, ws = interpolants(learner.grid, state)
        learner.targets[tidx] = dot(learner.values[tidx, inds], ws)
    end
    return learner.targets
end

function predict(learner::TDLearner, states::Array{Float64})
    state_dim, num_states = size(states)
    values = zeros(learner.target_dim, num_states)
    for sidx in 1:num_states
        values[:, sidx] = predict(learner, states[:, sidx])
    end
    return values
end

function learn(learner::TDLearner, x::Array{Float64}, r::Array{Float64}, 
        nx::Array{Float64}, done::Bool)

    # update 
    total_td_error = 0
    inds, ws = interpolants(learner.grid, x)
    for (ind, w) in zip(inds, ws)
        # target value
        target = r
        if !done
            target += learner.discount * predict(learner, x)
        end

        # update
        td_error = w * (target - predict(learner, nx))
        learner.values[:, ind] += learner.lr * td_error
        total_td_error += td_error
    end

    # store td-error associated with this state for later use as feedback
    push!(learner.td_errors, sum(abs(total_td_error)))
end

function learn(learner::TDLearner, experience::ExperienceMemory)
    for i in 1:length(experience)
        x, a, r, nx, done = get(experience, i)
        learn(learner, x, a, r, nx, done)
    end
    return get_feedback(learner)
end
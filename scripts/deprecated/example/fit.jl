# budget, env, agent, learner, t_max, mc_runs, discount 
function fit(learner::PredictionModel, env::Env, agent::Agent; 
            budget::Float64 = 5., t_max::Int = 10, mc_runs::Int = 1, discount::Float64 = 1., 
        states::Array{Float64} = zeros(1), true_v = nothing, ncurve_pts::Int = 10)
    timer = BudgetTimer(budget)
    curve = Float64[]
    curve_pt = .6
    while has_time_remaining(timer)
        
        # collect data
        init_x = reset(env)
        x = copy(init_x)
        for run in 1:mc_runs
            copy!(x, init_x)
            reset(env, x)
            t, done = 0, false
            while t < t_max && !done
                a = step(agent, x)
                nx, r, done = step(env, a)
                learner_done = AutoRisk.step(learner, x, a, [r], nx, done)
                done = done || learner_done
                x = nx
                t += 1
            end
        end
        
        # feedback
        feedback = AutoRisk.get_feedback(learner)
        update!(env, feedback)
        
        # track progress if ground truth available
        if !(true_v == nothing) && past_fraction(timer, curve_pt / ncurve_pts)
            pause(timer)
            pred_v = predict(learner, states)
            push!(curve, mean(sqrt((true_v - pred_v).^2)))
            curve_pt += 1.
            unpause(timer)
        end
    end
    learner, curve
end
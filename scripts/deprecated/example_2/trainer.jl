
"""
# Description:
    - Trainer implements the algorithm for collecting experience on which 
        a learner should learn. It is also responsible for monitoring learning 
        progress. In particular, a trainer should implement the following:
            + incorporate_feedback
            + update_experience
            + finalize_experience
            + monitor_progress
"""
abstract Trainer

finished_training(trainer::Trainer) = has_time_remaining(trainer.monitor.timer)
prepare_experience(trainer::Trainer, env::Env, policy::Policy) = trainer.experience

function update_experience(trainer::Trainer, x::Array{Float64}, 
        a::Array{Float64}, r::Array{Float64}, nx::Array{Float64}, done::Bool)
    trainer.step_count += 1
    update_experience(trainer.experience, x, a, r, nx, done)
end

function collect_experience(trainer::Trainer, env::Env, policy::Policy)
    ix = reset!(env, trainer.initial_state_dist)
    for run in 1:trainer.num_mc_runs
        x = ix[:]
        reset!(env, x)
        done = false
        for t in 1:trainer.max_episode_steps
            a = step(policy, x)
            nx, r, done, info = step(env, a)
            update_experience(trainer, x, a, r, nx, done)
            x = nx
            if done break end
        end
    end
end

function incorporate_feedback(trainer::Trainer, feedback::Array{Float64})
    reset_experience(trainer.experience)
end

function run_training_step(trainer::Trainer, learner::Learner, env::Env, policy::Policy)
    collect_experience(trainer, env, policy)
    feedback = learn(learner, prepare_experience(trainer, env, policy))
    incorporate_feedback(trainer, feedback)
end

function monitor_progress(trainer::Trainer, learner::Learner, env::Env, policy::Policy)
    monitor_progress(trainer.monitor, learner)
end

function train(trainer::Trainer, learner::Learner, env::Env, policy::Policy)
    while !finished_training(trainer)
        run_training_step(trainer, learner, env, policy)
        monitor_progress(trainer, learner, env, policy)
    end
end

function reset!(trainer::Trainer)
    reset_experience(trainer.experience)
end

type AdaptiveTrainer <: Trainer
    initial_state_dist::Distribution
    monitor::TrainingMonitor
    update_dist_freq::Int
    max_episode_steps::Int
    num_mc_runs::Int
    experience::ExperienceMemory
    step_count::Int
    prev_update_step::Int
    function AdaptiveTrainer(initial_state_dist::Distribution,
            monitor::TrainingMonitor;
            update_dist_freq::Int = 100,
            max_episode_steps::Int = 1,
            num_mc_runs::Int = 1)
        return new(initial_state_dist, monitor, update_dist_freq, 
            max_episode_steps, num_mc_runs, reset_experience(), 0, 0)
    end
end

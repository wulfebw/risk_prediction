
type TrainingMonitor
    timer::BudgetTimer
    eval_states::Array{Float64}
    v_true::Array{Float64}
    info::Dict{String, Any}
end

is_on(monitor::TrainingMonitor) = return length(v_true) > 0
log_value(monitor::TrainingMonitor, k::String, v::Any) = monitor.info[k] = v

function monitor_progress(monitor::TrainingMonitor, learner::Learner)
    pause(trainer.monitor.timer)
    if is_on(trainer.monitor)
        v_pred = predict(learner, monitor.eval_states)
        loss = rmse(monitor.v_true, v_pred)
        log_value(monitor, "state-value rmse loss", loss)
    end
    unpause(trainer.monitor.timer)
end
    
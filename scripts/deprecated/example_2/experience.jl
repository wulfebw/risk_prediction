
typealias ExperienceType Union{Array{Float64}, Float64, Bool}
typealias ExperienceMemory Dict{String,Array{ExperienceType}}

length(mem::ExperienceMemory) = length(mem["x"])

function reset_experience(d::ExperienceMemory = ExperienceMemory())
    for k in ["x", "a", "r", "nx", "done"]
        d[k] = ExperienceType[]
    end
    return d
end

function update_experience(experience::ExperienceMemory, x::Array{Float64}, 
        a::Array{Float64}, r::Array{Float64}, nx::Array{Float64}, done::Bool)
    push!(experience["x"], x)
    push!(experience["a"], a)
    if typeof(r) == Float64 
        r = [r] 
    end
    push!(experience["r"], r)
    push!(experience["nx"], nx)
    push!(experience["done"], done)
end

function get(experience::ExperienceMemory, index::Int)
    assert index <= length(experience)
    x = experience["x"][i]
    a = experience["a"][i]
    r = experience["r"][i]
    nx = experience["nx"][i]
    done = experience["done"][i]
    return x, a, r, nx, done
end
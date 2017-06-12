
using Base.Test

include("testing_params.jl")

"""
Include all files that start with "test" recursively.
Test files should call their own tests.
"""
function load_tests(path::String)
    @assert isdir(path)
    for path in readdir(path)
        if isfile(path) && startswith(path, "test_")
            include(path)   
        elseif isdir(path) && startswith(path, "test_")
            load_tests(path)
        end
    end
end

load_tests(".")
println("All tests pass!")
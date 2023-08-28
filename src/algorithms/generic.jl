@concrete struct NonlinearReformulation{method} <: AbstractComplementarityAlgorithm
    nlsolver
    diffmode
end

@truncate_stacktrace NonlinearReformulation 1

function NonlinearReformulation(method::Symbol=:smooth, nlsolver=NewtonRaphson(),diffmode = AutoSparseFiniteDiff())
    return NonlinearReformulation{method}(nlsolver,diffmode)
end

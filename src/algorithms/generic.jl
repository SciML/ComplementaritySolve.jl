@concrete struct NonlinearReformulation{method} <: AbstractComplementarityAlgorithm
    nlsolver
end

@truncate_stacktrace NonlinearReformulation 1

function NonlinearReformulation(method::Symbol=:smooth, nlsolver=DEFAULT_NLSOLVER)
    return NonlinearReformulation{method}(nlsolver)
end

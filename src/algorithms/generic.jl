@concrete struct NonlinearReformulation{method} <: AbstractComplementarityAlgorithm
    nlsolver
end


function NonlinearReformulation(method::Symbol = :smooth, nlsolver = DEFAULT_NLSOLVER)
    return NonlinearReformulation{method}(nlsolver)
end

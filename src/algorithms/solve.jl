function solve(prob::AbstractComplementarityProblem,
    args...;
    sensealg=nothing,
    u0=nothing,
    p=nothing,
    kwargs...)
    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p
    sensealg = sensealg === nothing ? __default_sensealg(prob) : sensealg
    return __solve(prob, sensealg, u0, p, args...; kwargs...)
end

function solve(prob::Union{LinearComplementarityProblem, MixedLinearComplementarityProblem},
    args...;
    sensealg=nothing,
    u0=nothing,
    M=nothing,
    q=nothing,
    kwargs...)
    u0 = u0 !== nothing ? u0 : prob.u0
    M = M !== nothing ? M : prob.M
    q = q !== nothing ? q : prob.q
    sensealg = sensealg === nothing ? __default_sensealg(prob) : sensealg
    return __solve(prob, sensealg, u0, M, q, args...; kwargs...)
end

function __default_sensealg(::T) where {T <: AbstractComplementarityProblem}
    @warn "No default sensealg for type $(T). Please specify a sensealg if using adjoints." maxlog=1
    return nothing
end
__default_sensealg(::LinearComplementarityProblem) = LinearComplementarityAdjoint()
__default_sensealg(::MixedComplementarityProblem) = MixedComplementarityAdjoint()

# Algorithms should dispatch on __solve
function __solve end

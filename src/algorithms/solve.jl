function solve(prob::AbstractComplementarityProblem,
    args...;
    sensealg=nothing,
    u0=nothing,
    p=nothing,
    kwargs...)
    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p
    sensealg = sensealg === nothing ? __default_sensealg(prob) : sensealg
    solver, args_ = __solver_and_args(prob, args...)
    return __solve(prob, sensealg, solver, u0, p, args_...; kwargs...)
end

function solve(prob::AbstractLinearComplementarityProblem,
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
    solver, args_ = __solver_and_args(prob, args...)
    return __solve(prob, sensealg, solver, u0, M, q, args_...; kwargs...)
end

function __solver_and_args(prob, args...)
    return length(args) == 0 ? (__default_solver(prob), ()) : (first(args), args[2:end])
end

function __default_sensealg(::T) where {T <: AbstractComplementarityProblem}
    @warn "No default sensealg for type $(T). Please specify a sensealg if using \
           adjoints." maxlog=1
    return nothing
end
__default_sensealg(::LinearComplementarityProblem) = LinearComplementarityAdjoint()
__default_sensealg(::MixedComplementarityProblem) = MixedComplementarityAdjoint()

function __default_solver(::T) where {T <: AbstractComplementarityProblem}
    return error("No default solver for type $(T). Please specify a solver.")
end
__default_solver(::LinearComplementarityProblem) = NonlinearReformulation()
__default_solver(::MixedComplementarityProblem) = NonlinearReformulation()

# Algorithms should dispatch on __solve
function __solve end

function __solve(prob::AbstractComplementarityProblem,
    sensealg::Union{Nothing, AbstractComplementaritySensitivityAlgorithm},
    args...;
    kwargs...)
    return __solve(prob, args...; kwargs...)
end

## Dispatch only if using SensitivityAlgorithm else differentiate through the solve
function CRC.rrule(::typeof(__solve),
    prob::AbstractComplementarityProblem,
    sensealg::AbstractComplementaritySensitivityAlgorithm,
    solver,
    args...;
    kwargs...)
    sol = __solve(prob, solver, args...; kwargs...)
    function ∇__solve(∂sol)
        ∂p = __solve_adjoint(prob, sensealg, sol, ∂sol, args...; kwargs...)
        return (∂∅, ∂∅, ∂∅, ∂∅, ∂∅, ∂p...)
    end
    return sol, ∇__solve
end

function solve(
        prob::AbstractComplementarityProblem, args...; sensealg = nothing, u0 = nothing,
        p = nothing, kwargs...
    )
    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p
    sensealg = sensealg === nothing ? __default_sensealg(prob) : sensealg
    solver, args_ = __solver_and_args(prob, args...)
    return __solve(prob, sensealg, solver, u0, p, args_...; kwargs...)
end

function solve(
        prob::AbstractLinearComplementarityProblem, args...; sensealg = nothing,
        u0 = nothing, M = nothing, q = nothing, kwargs...
    )
    u0 = u0 !== nothing ? u0 : prob.u0
    M = M !== nothing ? M : prob.M
    q = q !== nothing ? q : prob.q
    sensealg = sensealg === nothing ? __default_sensealg(prob) : sensealg
    solver, args_ = __solver_and_args(prob, args...)
    return __solve(prob, sensealg, solver, u0, M, q, args_...; kwargs...)
end

"""
    solve(prob::AbstractComplementaritySystem, alg::AbstractComplementaritySystemAlgorithm; kwargs...)

Solve a complementarity system with a compatible system algorithm.

# Arguments

- `prob`: A complementarity system implementing the
  `AbstractComplementaritySystem` field contract.
- `alg`: A compatible `AbstractComplementaritySystemAlgorithm` implementation.

# Keyword Arguments

- `kwargs...`: Solver-specific keyword arguments forwarded unchanged to the
  developer `__solve` implementation.

# Interface

Packages extending this interface must implement
`ComplementaritySolve.__solve(prob, alg; kwargs...)`. The generic `solve` function
does not inspect concrete system fields, so the extension owns validation and the
returned continuous-solver solution type.
"""
function solve(
        prob::AbstractComplementaritySystem, alg::AbstractComplementaritySystemAlgorithm;
        kwargs...
    )
    return __solve(prob, alg; kwargs...)
end

function __solver_and_args(prob, args...)
    return length(args) == 0 ? (__default_solver(prob), ()) : (first(args), args[2:end])
end

function __default_sensealg(::T) where {T <: AbstractComplementarityProblem}
    @warn "No default sensealg for type $(T). Please specify a sensealg if using \
           adjoints." maxlog = 1
    return nothing
end
__default_sensealg(::LCP) = LinearComplementarityAdjoint()
__default_sensealg(::MCP) = MixedComplementarityAdjoint()

function __default_solver(::T) where {T <: AbstractComplementarityProblem}
    return error("No default solver for type $(T). Please specify a solver.")
end
## Defaulting to SimpleNewtonRaphson() since it is the most robust
## and works well with inplace/out of place and also works OOTB with GPUs
__default_solver(::Union{LCP, MCP}) = NonlinearReformulation(:smooth, DEFAULT_NLSOLVER)

"""
    __solve(prob, alg, args...; kwargs...)

Developer extension hook implementing a complementarity solve.

# Arguments

- `prob`: A subtype of `AbstractComplementarityProblem` or
  `AbstractComplementaritySystem`.
- `alg`: A compatible developer algorithm subtype.
- `args...`: Normalized solve data forwarded by `solve`; problem algorithms receive
  the effective initial state and problem data, while system algorithms receive no
  additional positional data.

# Keyword Arguments

- `kwargs...`: Algorithm-specific solve options forwarded from `solve`.

# Returns

- A solution satisfying the documented solution contract for the problem family, or
  the underlying continuous-solver solution for complementarity systems.

# Interface

This qualified, non-exported name is the stable extension point for solver packages.
Implementations must not mutate stored problem data when processing per-call
overrides, and must preserve the original `prob` and `alg` in complementarity solution
objects. Applications should call `solve`, not `__solve`, directly.
"""
function __solve end

"""
    __solve_adjoint(prob, sensealg, sol, Δsol, args...; kwargs...)

Developer extension hook for the reverse rule of a complementarity solve.

# Arguments

- `prob`: The original complementarity problem.
- `sensealg`: An `AbstractComplementaritySensitivityAlgorithm` compatible with
  `prob` and `sol`.
- `sol`: The primal complementarity solution.
- `Δsol`: The incoming tangent for `sol`.
- `args...`: The normalized primal solve data passed to `__solve`.

# Keyword Arguments

- `kwargs...`: The solve keyword arguments supplied to the primal solve.

# Returns

- Tangents in the order of the differentiable primal solve data.

# Interface

Implementations must return `NoTangent` or `ZeroTangent` for nondifferentiable data
and shape-compatible tangents for differentiable data. This hook is called by the
`ChainRulesCore.rrule` for `__solve`; applications should select a concrete sensitivity
algorithm through `solve(...; sensealg = ...)` instead of invoking it directly.
"""
function __solve_adjoint end

function __solve(
        prob::AbstractComplementarityProblem,
        sensealg::Union{Nothing, AbstractComplementaritySensitivityAlgorithm}, args...;
        kwargs...
    )
    return __solve(prob, args...; kwargs...)
end

## Dispatch only if using SensitivityAlgorithm else differentiate through the solve
function CRC.rrule(
        ::typeof(__solve), prob::AbstractComplementarityProblem,
        sensealg::AbstractComplementaritySensitivityAlgorithm, solver, args...; kwargs...
    )
    sol = __solve(prob, solver, args...; kwargs...)
    function ∇__solve(∂sol)
        ∂p = __solve_adjoint(prob, sensealg, sol, ∂sol, args...; kwargs...)
        return (∂∅, ∂∅, ∂∅, ∂∅, ∂∅, ∂p...)
    end
    return sol, ∇__solve
end

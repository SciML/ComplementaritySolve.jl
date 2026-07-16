"""
    solve(prob::AbstractComplementarityProblem[, alg]; sensealg = nothing,
        u0 = nothing, p = nothing, kwargs...)
    solve(prob::AbstractLinearComplementarityProblem[, alg]; sensealg = nothing,
        u0 = nothing, M = nothing, q = nothing, kwargs...)
    solve(prob::LinearComplementaritySystem, alg::NaiveLCSAlgorithm;
        ode_kwargs = (;), lcp_kwargs = (;), kwargs...)

Solve a complementarity problem or linear complementarity system.

The optional `alg` argument selects a ComplementaritySolve algorithm. When no
algorithm is supplied, `LinearComplementarityProblem` and
`MixedComplementarityProblem` use the default nonlinear reformulation solver.

# Arguments

- `prob`: Complementarity problem or system to solve.
- `alg`: Optional solver algorithm, such as `PGS()`, `RPSOR()`,
    `NonlinearReformulation()`, or `NaiveLCSAlgorithm(...)`.

# Keywords

- `sensealg`: Sensitivity algorithm used for adjoint rules. Defaults to the
    problem-specific sensitivity algorithm when one exists.
- `u0`, `p`: Override the initial state and parameters stored in `prob`.
- `M`, `q`: Override the matrix and vector of a linear complementarity problem.
- `ode_kwargs`: Named tuple forwarded to the ODE or steady-state solve for
    `LinearComplementaritySystem`s.
- `lcp_kwargs`: Named tuple forwarded to the embedded LCP solve for
    `LinearComplementaritySystem`s.
- `kwargs...`: Additional solver keywords forwarded to the selected algorithm.

# Examples

```julia
using ComplementaritySolve

M = [2.0 -1.0; -1.0 2.0]
q = [-1.0, -1.0]
prob = LCP(M, q, zeros(2))

sol = solve(prob, PGS(); abstol = 1e-8)
sol.u
```
"""
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

# Algorithms should dispatch on __solve
function __solve end
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

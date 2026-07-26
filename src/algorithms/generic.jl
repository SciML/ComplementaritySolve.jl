"""
    NonlinearReformulation(method::Symbol = :smooth, nlsolver = DEFAULT_NLSOLVER)

Solve complementarity problems by reformulating the complementarity condition as a
nonlinear residual and delegating to a NonlinearSolve.jl-compatible solver.

# Arguments

- `method`: Reformulation to use. Supported values are `:smooth` and `:minmax`.
- `nlsolver`: Nonlinear solver algorithm passed to `NonlinearSolve.solve`.

# Fields

- `nlsolver`: Solver used on the transformed nonlinear problem.

# Examples

```julia
using ComplementaritySolve

prob = LinearComplementarityProblem([2.0 -1.0; -1.0 2.0], [-1.0, -1.0])
sol = ComplementaritySolve.solve(prob, NonlinearReformulation(:smooth))
```
"""
@concrete struct NonlinearReformulation{method} <: AbstractComplementarityAlgorithm
    nlsolver
end


function NonlinearReformulation(method::Symbol = :smooth, nlsolver = DEFAULT_NLSOLVER)
    return NonlinearReformulation{method}(nlsolver)
end

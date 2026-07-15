"""
    AbstractComplementaritySolution

Developer interface for complementarity solution objects.

# Interface

- `sol.u` must contain the computed complementarity variables.
- `sol.residual` must contain the solver residual or residual summary.
- `sol.prob` must reference the problem that was solved.
- `sol.alg` must reference the algorithm used for the solve.
- `sol.retcode` must be a `SciMLBase.ReturnCode` value.
"""
abstract type AbstractComplementaritySolution end

"""
    AbstractLinearComplementaritySolution <: AbstractComplementaritySolution

Developer interface for solutions of linear complementarity problems.

Subtypes follow the [`AbstractComplementaritySolution`](@ref) field contract and
are returned by algorithms solving `AbstractLinearComplementarityProblem`s.
"""
abstract type AbstractLinearComplementaritySolution <: AbstractComplementaritySolution end

function Base.show(io::IO, m::MIME"text/plain", A::AbstractComplementaritySolution)
    println(io, string(nameof(typeof(A)), " with retcode: ", A.retcode))
    print(io, "u: ")
    show(io, m, A.u)
    return nothing
end

"""
    LinearComplementaritySolution(u, residual, prob, alg, retcode)

Solution object returned by linear complementarity problem solvers.

# Fields

- `u`: Computed complementarity variable.
- `residual`: Solver-specific residual or residual summary.
- `prob`: Original complementarity problem.
- `alg`: Algorithm used to solve the problem.
- `retcode`: `SciMLBase.ReturnCode` describing solver termination.
"""
@concrete struct LinearComplementaritySolution <: AbstractLinearComplementaritySolution
    u
    residual
    prob
    alg
    retcode::ReturnCode.T
end

"""
    MixedComplementaritySolution(u, residual, prob, alg, retcode)

Solution object returned by mixed and nonlinear complementarity problem solvers.

# Fields

- `u`: Computed complementarity variable.
- `residual`: Solver-specific residual or residual summary.
- `prob`: Original complementarity problem.
- `alg`: Algorithm used to solve the problem.
- `retcode`: `SciMLBase.ReturnCode` describing solver termination.
"""
@concrete struct MixedComplementaritySolution <: AbstractComplementaritySolution
    u
    residual
    prob
    alg
    retcode::ReturnCode.T
end

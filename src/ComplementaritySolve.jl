module ComplementaritySolve

# FIXME: Some of these dependencies are not needed and can be reorganized
# But during research stages let's keep them all here
# Before release we will clean things up

## Core / QOL Dependencies
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent
using GPUArraysCore: GPUArraysCore
using SciMLBase: SciMLBase, LinearProblem, NonlinearFunction,
    NonlinearProblem, ODEFunction, ODEProblem, ReturnCode,
    SteadyStateProblem, isinplace
using SciMLPublic: @public
using CommonSolve: CommonSolve
using ConcreteStructs: ConcreteStructs, @concrete
using DifferentiationInterface: DifferentiationInterface, AutoForwardDiff, AutoZygote,
    jacobian
## Stdlibs
using LinearAlgebra: LinearAlgebra, Diagonal, I, diagind, mul!, norm, pinv, \, /
using SparseArrays: SparseArrays
## SciML Dependencies
using LinearSolve: LinearSolve
using SciMLOperators: FunctionOperator
using SimpleNonlinearSolve: SimpleNonlinearSolve, SimpleNewtonRaphson
using NonlinearSolve: NonlinearSolve
## AD Packages (for sensitivities & PATHSolver; move to extensions)
using ForwardDiff: ForwardDiff
using Zygote: Zygote
## External Solvers (for PATHSolver; move to extensions)
using PATHSolver: PATHSolver
## Fast Batching Support
using NNlib: NNlib, batched_mul, batched_mul!, batched_transpose, ⊠
using Polyester: Polyester, @batch

import CommonSolve: init, solve, solve!
import ChainRulesCore as CRC

const ∂0 = ZeroTangent()
const ∂∅ = NoTangent()
const ∅p = SciMLBase.NullParameters()
const AA = AbstractArray
const AV = AbstractVector
const AM = AbstractMatrix
const AA3 = AbstractArray{T, 3} where {T}

const DEFAULT_NLSOLVER = SimpleNewtonRaphson()

"""
    AbstractComplementarityAlgorithm

Developer interface for complementarity problem algorithms.

Algorithm types should subtype `AbstractComplementarityAlgorithm` when they are
intended to be passed as the second argument to `solve(prob, alg; kwargs...)` for
`AbstractComplementarityProblem`s.

# Interface

- Implement `ComplementaritySolve.__solve(prob, alg, args...; kwargs...)` for each
    supported problem family.
- Return an `AbstractComplementaritySolution` subtype whose `prob` and `alg` fields
    reference the original problem and algorithm.
- Accept solver keywords through `kwargs...` when the wrapped numerical method
    supports them.

This is a developer extension point. User code should prefer the concrete algorithm
constructors exported by ComplementaritySolve.
"""
abstract type AbstractComplementarityAlgorithm end

"""
    AbstractComplementaritySystemAlgorithm

Developer interface for algorithms that solve complementarity systems.

# Interface

- Implement `ComplementaritySolve.__solve(prob, alg; kwargs...)`.
- Return the solution object produced by the wrapped ODE or steady-state solve.
- Forward relevant solver keywords to the continuous dynamics solve and the embedded
    complementarity solve.
"""
abstract type AbstractComplementaritySystemAlgorithm end

"""
    AbstractComplementaritySensitivityAlgorithm

Developer interface for adjoint rules for complementarity solves.

# Interface

- Implement `__solve_adjoint(prob, sensealg, sol, ∂sol, args...; kwargs...)`.
- Return parameter/data tangents compatible with the corresponding `solve` call.
- Use only the public problem fields documented by the concrete problem type.

This interface is used by the ChainRulesCore rule for differentiating through
`solve(prob, alg; sensealg)`.
"""
abstract type AbstractComplementaritySensitivityAlgorithm end

include("utils.jl")

include("problems/complementarity_problems.jl")
include("problems/complementarity_systems.jl")

include("algorithms/solve.jl")
include("algorithms/generic.jl")
include("algorithms/lcp/nonlinear_reformulation.jl")
include("algorithms/lcp/bokhoven_iterative.jl")
include("algorithms/lcp/rpsor.jl")
include("algorithms/lcp/ipm.jl")
include("algorithms/lcp/fallback.jl")
include("algorithms/mcp/nonlinear_reformulation.jl")
include("algorithms/mcp/pathsolver.jl")
include("algorithms/lcs/naive_lcs.jl")

include("solutions.jl")

include("sensitivity/lcp.jl")
include("sensitivity/mcp.jl")

export LinearComplementarityProblem, MixedLinearComplementarityProblem,
    NonlinearComplementarityProblem, MixedComplementarityProblem
export LinearComplementaritySystem
export LCP, MLCP, NCP, MCP, LCS  # Short aliases
export BokhovenIterativeAlgorithm,
    NonlinearReformulation, RPSOR, PGS, PSOR, RPGS, InteriorPointMethod
export PATHSolverAlgorithm
export NaiveLCSAlgorithm
export LinearComplementarityAdjoint, MixedComplementarityAdjoint
export LinearComplementaritySolution, MixedComplementaritySolution

@public AbstractComplementarityAlgorithm, AbstractComplementaritySystemAlgorithm
@public AbstractComplementaritySensitivityAlgorithm
@public AbstractComplementarityProblem, AbstractLinearComplementarityProblem
@public AbstractNonlinearComplementarityProblem, AbstractComplementaritySystem
@public AbstractComplementaritySolution, AbstractLinearComplementaritySolution
@public isbatched, __solve, __solve_adjoint

include("precompilation.jl")

end

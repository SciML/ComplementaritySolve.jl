module ComplementaritySolve

# FIXME: Some of these dependencies are not needed and can be reorganized
# But during research stages let's keep them all here
# Before release we will clean things up

## Core / QOL Dependencies
using ArrayInterfaceCore, ChainRulesCore, CommonSolve, ConcreteStructs, SciMLBase
## Stdlibs
using LinearAlgebra, Markdown, SparseArrays
## SciML Dependencies
using LinearSolve, SciMLOperators, SimpleNonlinearSolve, NonlinearSolve
## AD Packages (for sensitivities & PATHSolver; move to extensions)
using ForwardDiff, Zygote
## External Solvers (for PATHSolver; move to extensions)
using PATHSolver
## Fast Batching Support
using NNlib, Polyester

import CommonSolve: init, solve, solve!
import ChainRulesCore as CRC
import FillArrays: AbstractFill
import TruncatedStacktraces: @truncate_stacktrace

const ∂0 = ZeroTangent()
const ∂∅ = NoTangent()
const ∅p = SciMLBase.NullParameters()
const AA = AbstractArray
const AV = AbstractVector
const AM = AbstractMatrix
const AA3 = AbstractArray{T, 3} where {T}

### ----- Type Piracy Starts ----- ###
ArrayInterfaceCore.can_setindex(::Type{<:AbstractFill}) = false
ArrayInterfaceCore.can_setindex(::Zygote.OneElement) = false
### ------ Type Piracy Ends ------ ###

abstract type AbstractComplementarityAlgorithm end
abstract type AbstractComplementaritySystemAlgorithm end
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

export LinearComplementarityProblem,
    MixedLinearComplementarityProblem,
    NonlinearComplementarityProblem,
    MixedComplementarityProblem
export LinearComplementaritySystem
export LCP, MLCP, NCP, MCP, LCS  # Short aliases
export BokhovenIterativeAlgorithm,
    NonlinearReformulation, RPSOR, PGS, PSOR, RPGS, InteriorPointMethod
export PATHSolverAlgorithm
export NaiveLCSAlgorithm
export LinearComplementarityAdjoint, MixedComplementarityAdjoint
export LinearComplementaritySolution, MixedComplementaritySolution
export solve

end

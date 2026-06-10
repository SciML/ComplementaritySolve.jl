module ComplementaritySolve

# FIXME: Some of these dependencies are not needed and can be reorganized
# But during research stages let's keep them all here
# Before release we will clean things up

## Core / QOL Dependencies
using ArrayInterfaceCore: ArrayInterfaceCore
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent
using GPUArraysCore: GPUArraysCore
using SciMLBase: SciMLBase, FunctionOperator, LinearProblem, NonlinearFunction,
    NonlinearProblem, ODEFunction, ODEProblem, ReturnCode,
    SteadyStateProblem, isinplace
using CommonSolve: CommonSolve
using ConcreteStructs: ConcreteStructs, @concrete
## Stdlibs
using LinearAlgebra: LinearAlgebra, Diagonal, I, diagind, mul!, norm, pinv, \, /
using Markdown: Markdown, @doc_str
using SparseArrays: SparseArrays
## SciML Dependencies
using LinearSolve: LinearSolve
## SciMLOperators is used transitively via SciMLBase (FunctionOperator)
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
import FillArrays: AbstractFill
import TruncatedStacktraces: @truncate_stacktrace

const ∂0 = ZeroTangent()
const ∂∅ = NoTangent()
const ∅p = SciMLBase.NullParameters()
const AA = AbstractArray
const AV = AbstractVector
const AM = AbstractMatrix
const AA3 = AbstractArray{T, 3} where {T}

const DEFAULT_NLSOLVER = SimpleNewtonRaphson()

### ----- Type Piracy Starts ----- ###
ArrayInterfaceCore.can_setindex(::Type{<:AbstractFill}) = false
ArrayInterfaceCore.can_setindex(::Zygote.OneElement) = false

# ForwardDiff 1.x seeds dual arrays with scalar `setindex!` loops (via
# `structural_eachindex`), which errors on GPU arrays with scalar indexing
# disallowed. ForwardDiff 0.10 used broadcast and worked on GPU arrays, so
# restore broadcast-based seeding for them. TODO: upstream to ForwardDiff as a
# GPUArraysCore extension.
function ForwardDiff.seed!(
        duals::GPUArraysCore.AbstractGPUArray{ForwardDiff.Dual{T, V, N}}, x,
        seed::ForwardDiff.Partials{N, V} = zero(ForwardDiff.Partials{N, V})
    ) where {T, V, N}
    duals .= ForwardDiff.Dual{T, V, N}.(x, Ref(seed))
    return duals
end

function ForwardDiff.seed!(
        duals::GPUArraysCore.AbstractGPUArray{ForwardDiff.Dual{T, V, N}}, x,
        seeds::NTuple{N, ForwardDiff.Partials{N, V}}
    ) where {T, V, N}
    dual_inds = 1:N
    duals[dual_inds] .= ForwardDiff.Dual{T, V, N}.(view(x, dual_inds), seeds)
    return duals
end

function ForwardDiff.seed!(
        duals::GPUArraysCore.AbstractGPUArray{ForwardDiff.Dual{T, V, N}}, x, index,
        seed::ForwardDiff.Partials{N, V} = zero(ForwardDiff.Partials{N, V})
    ) where {T, V, N}
    dual_inds = index:length(duals)
    duals[dual_inds] .= ForwardDiff.Dual{T, V, N}.(view(x, dual_inds), Ref(seed))
    return duals
end

function ForwardDiff.seed!(
        duals::GPUArraysCore.AbstractGPUArray{ForwardDiff.Dual{T, V, N}}, x, index,
        seeds::NTuple{N, ForwardDiff.Partials{N, V}}, chunksize = N
    ) where {T, V, N}
    offset = index - 1
    dual_inds = (1 + offset):(offset + chunksize)
    duals[dual_inds] .= ForwardDiff.Dual{T, V, N}.(view(x, dual_inds), seeds[1:chunksize])
    return duals
end

### ------ Type Piracy Ends ------ ###
# NOTE: LinearSolve.defaultalg for AbstractSciMLOperator + AbstractGPUArray was upstreamed

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
export solve

include("precompilation.jl")

end

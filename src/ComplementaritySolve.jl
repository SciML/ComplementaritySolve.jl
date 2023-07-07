module ComplementaritySolve

# FIXME: Some of these dependencies are not needed and can be reorganized
# But during research stages let's keep them all here
# Before release we will clean things up
using ArrayInterfaceCore,
    ChainRulesCore,
    CommonSolve,
    ComponentArrays, # doesn't need a dependency on this (remove)
    ConcreteStructs,
    FillArrays, # only for Zygote (move to ext later)
    LinearAlgebra,
    LinearSolve, # doesn't need a dependency on this (users must load this though)
    Markdown,
    NonlinearSolve, # doesn't need a dependency on this (remove)
    NNlib, # batching support (move to ext later)
    Polyester, # batching support (move to ext later)
    SimpleNonlinearSolve, # doesn't need a dependency on this (remove)
    SciMLBase,
    SciMLOperators, # for MCP sensitivities (move to ext later)
    SparseArrays, # Can be dropped?
    Zygote # For MCP sensitivities (move to ext later)
import CommonSolve: init, solve, solve!
import ChainRulesCore as CRC
import TruncatedStacktraces: @truncate_stacktrace

const ∂0 = ZeroTangent()
const ∂∅ = NoTangent()
const ∅p = SciMLBase.NullParameters()

### ----- Type Piracy Starts ----- ###
ArrayInterfaceCore.can_setindex(::Type{<:FillArrays.AbstractFill}) = false
ArrayInterfaceCore.can_setindex(::Zygote.OneElement) = false
### ------ Type Piracy Ends ------ ###

abstract type AbstractComplementarityAlgorithm end
abstract type AbstractComplementaritySystemAlgorithm end

include("utils.jl")

include("problems/complementarity_problems.jl")
include("problems/complementarity_systems.jl")

include("algorithms/generic.jl")
include("algorithms/lcp/nonlinear_reformulation.jl")
include("algorithms/lcp/bokhoven_iterative_lcp.jl")
include("algorithms/lcp/rpsor.jl")
include("algorithms/lcp/fallback.jl")
include("algorithms/mcp/nonlinear_reformulation.jl")
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
export BokhovenIterativeLCPAlgorithm, NonlinearReformulation, RPSOR, PGS, PSOR, RPGS
export NaiveLCSAlgorithm
export LinearComplementarityAdjoint, MixedComplementarityAdjoint
export LinearComplementaritySolution, MixedComplementaritySolution
export solve

end

module ComplementaritySolve

using ArrayInterfaceCore,
    ChainRulesCore,
    CommonSolve,
    ComponentArrays,
    FillArrays,
    LinearAlgebra,
    LinearSolve,
    NonlinearSolve,
    NNlib,
    SimpleNonlinearSolve,
    SciMLBase,
    SciMLOperators,
    SparseArrays,
    Zygote
import CommonSolve: init, solve, solve!
import ChainRulesCore as CRC
import TruncatedStacktraces: @truncate_stacktrace

const ∂0 = ZeroTangent()
const ∂∅ = NoTangent()
const ∅p = SciMLBase.NullParameters()

# Needs upstreaming
ArrayInterfaceCore.can_setindex(::Type{<:FillArrays.AbstractFill}) = false

include("utils.jl")
include("problems.jl")
include("algorithms.jl")
include("solutions.jl")
include("adjoint.jl")

export LinearComplementarityProblem,
    MixedLinearComplementarityProblem,
    NonlinearComplementarityProblem,
    MixedComplementarityProblem
export BokhovenIterativeLCPAlgorithm, NonlinearReformulation, RPSOR, PGS, PSOR, RPGS
export LinearComplementarityAdjoint
export LinearComplementaritySolution, MixedComplementaritySolution
export solve

end

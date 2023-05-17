module ComplementaritySolve

using ArrayInterfaceCore, CommonSolve, NonlinearSolve, SimpleNonlinearSolve, LinearSolve,
      FillArrays, ComponentArrays, LinearAlgebra, SparseArrays, ChainRulesCore
import CommonSolve: init, solve, solve!
import ChainRulesCore as CRC

const ∂0 = ZeroTangent()
const ∂∅ = NoTangent()

# Needs upstreaming
ArrayInterfaceCore.can_setindex(::Type{<:FillArrays.AbstractFill}) = false

include("utils.jl")
include("problems.jl")
include("algorithms.jl")
include("solutions.jl")
include("adjoint.jl")

export LinearComplementarityProblem, NonlinearComplementarityProblem,
       MixedComplementarityProblem
export BokhovenIterativeLCPAlgorithm, NonlinearReformulation
export LinearComplementarityAdjoint
export solve

end
module ComplementaritySolve

using ArrayInterfaceCore, CommonSolve, NonlinearSolve, SimpleNonlinearSolve, LinearSolve,
      FillArrays, ComponentArrays, LinearAlgebra, SparseArrays, ChainRulesCore,
      SciMLOperators
using Zygote
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

export LinearComplementarityProblem, MixedLinearComplementarityProblem,
       NonlinearComplementarityProblem, MixedComplementarityProblem
export BokhovenIterativeLCPAlgorithm, NonlinearReformulation
export LinearComplementarityAdjoint
export LinearComplementaritySolution, MixedComplementaritySolution
export solve

end
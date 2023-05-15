module ComplementaritySolve

using CommonSolve, NonlinearSolve, SimpleNonlinearSolve, LinearSolve
using ComponentArrays, LinearAlgebra, SparseArrays, ChainRulesCore
import CommonSolve: init, solve, solve!
import ChainRulesCore as CRC

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
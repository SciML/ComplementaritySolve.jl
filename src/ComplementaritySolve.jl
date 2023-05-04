module ComplementaritySolve

using CommonSolve, NonlinearSolve, SimpleNonlinearSolve
import CommonSolve: init, solve, solve!

include("utils.jl")
include("problems.jl")
include("solutions.jl")
include("algorithms.jl")

export LinearComplementarityProblem, NonlinearComplementarityProblem,
       MixedComplementarityProblem
export NonlinearReformulation
export solve

end
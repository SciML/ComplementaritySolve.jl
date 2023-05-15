abstract type AbstractComplementaritySolution end
abstract type AbstractLinearComplementaritySolution end

function Base.show(io::IO, sol::AbstractLinearComplementaritySolution)
    println(io, "Linear Complementarity Solution")
    println(io, "    z: ", sol.z)
    println(io, "    w: ", sol.w)
    return
end

struct LinearComplementaritySolution{zType, wType, residType,
                                     P <: AbstractComplementarityProblem,
                                     A <: AbstractComplementarityAlgorithm} <:
       AbstractLinearComplementaritySolution
    z::zType
    w::wType
    resid::residType
    prob::P
    alg::A
end

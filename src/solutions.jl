abstract type AbstractComplementaritySolution end
abstract type AbstractLinearComplementaritySolution end

function Base.show(io::IO, sol::AbstractLinearComplementaritySolution)
    println(io, "Linear Complementarity Solution")
    println(io, "    z: ", sol.z)
    println(io, "    w: ", sol.w)
    return
end

struct LinearComplementaritySolution{
    zType,
    wType,
    residType,
    P <: AbstractComplementarityProblem,
    A <: AbstractComplementarityAlgorithm,
} <: AbstractLinearComplementaritySolution
    z::zType
    w::wType
    resid::residType
    prob::P
    alg::A
end

struct MixedComplementaritySolution{
    uType,
    residType,
    P <: AbstractComplementarityProblem,
    A <: AbstractComplementarityAlgorithm,
} <: AbstractComplementaritySolution
    u::uType
    resid::residType
    prob::P
    alg::A
end

function Base.show(io::IO, sol::MixedComplementaritySolution)
    println(io, "Mixed Complementarity Solution")
    println(io, "    residual: ", sol.resid)
    println(io, "    u: ", sol.u)
    return
end

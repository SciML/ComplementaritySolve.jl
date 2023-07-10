abstract type AbstractComplementaritySolution end
abstract type AbstractLinearComplementaritySolution end

function Base.show(io::IO, sol::AbstractLinearComplementaritySolution)
    println(io, "Linear Complementarity Solution")
    println(io, "    z: ", sol.z)
    println(io, "    w: ", sol.w)
    return
end

@concrete struct LinearComplementaritySolution <: AbstractLinearComplementaritySolution
    z
    w
    resid
    prob
    alg
end

@concrete struct MixedComplementaritySolution <: AbstractComplementaritySolution
    u
    resid
    prob
    alg
end

function Base.show(io::IO, sol::MixedComplementaritySolution)
    println(io, "Mixed Complementarity Solution")
    println(io, "    residual: ", sol.resid)
    println(io, "    u: ", sol.u)
    return
end

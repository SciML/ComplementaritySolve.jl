abstract type AbstractComplementaritySolution end
abstract type AbstractLinearComplementaritySolution <: AbstractComplementaritySolution end

function Base.show(io::IO, m::MIME"text/plain", A::AbstractComplementaritySolution)
    println(io, string(nameof(typeof(A)), " with retcode: ", A.retcode))
    print(io, "u: ")
    show(io, m, A.u)
    return nothing
end

@concrete struct LinearComplementaritySolution <: AbstractLinearComplementaritySolution
    u
    residual
    prob
    alg
    retcode::ReturnCode.T
end

@concrete struct MixedComplementaritySolution <: AbstractComplementaritySolution
    u
    residual
    prob
    alg
    retcode::ReturnCode.T
end

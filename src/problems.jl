abstract type AbstractComplementarityProblem{iip} end

struct LinearComplementarityProblem{MType, qType, uType} <:
       AbstractComplementarityProblem{false}
    M::MType
    q::qType
    u0::uType
end

struct NonlinearComplementarityProblem{iip, F <: Function, uType, pType} <:
       AbstractComplementarityProblem{iip}
    f::F
    u0::uType
    p::pType
end

function NonlinearComplementarityProblem(prob::LinearComplementarityProblem)
    p = ComponentArray((; M, q))
    return NonlinearComplementarityProblem{false}((x, θ) -> θ.M * x + θ.q, prob.u0, p)
end

struct MixedComplementarityProblem{iip, F <: Function, uType, LB, UB, pType} <:
       AbstractComplementarityProblem{iip}
    f::F
    u0::uType
    lb::LB
    ub::UB
    p::pType
end

function MixedComplementarityProblem{iip}(f::F, u0::uType, lb::LB, ub::UB,
                                          p::pType=NullParameters()) where {iip, F, uType,
                                                                            LB, UB, pType}
    return MixedComplementarityProblem{iip, F, uType, LB, UB, pType}(f, u0, lb, ub, p)
end

abstract type AbstractComplementarityProblem{iip} end

struct LinearComplementarityProblem{MType, qType, uType} <:
       AbstractComplementarityProblem{false}
    M::MType
    q::qType
    u0::uType
end

struct NonlinearComplementarityProblem{iip, F <: Function, uType} <:
       AbstractComplementarityProblem{iip}
    f::F
    u0::uType
end

function NonlinearComplementarityProblem(prob::LinearComplementarityProblem)
    return NonlinearComplementarityProblem{false}((x, _) -> prob.q + prob.M * x, prob.u0)
end

struct MixedComplementarityProblem{iip, F <: Function, uType, LB, UB} <:
       AbstractComplementarityProblem{iip}
    f::F
    u0::uType
    lb::LB
    ub::UB
end

function MixedComplementarityProblem{iip}(f::F, u0::uType, lb::LB,
                                          ub::UB) where {iip, F, uType, LB, UB}
    return MixedComplementarityProblem{iip, F, uType, LB, UB}(f, u0, lb, ub)
end

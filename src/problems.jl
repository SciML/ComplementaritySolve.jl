abstract type AbstractComplementarityProblem{iip} end

struct LinearComplementarityProblem{MType, qType, uType} <:
       AbstractComplementarityProblem{false}
    M::MType
    q::qType
    u0::uType
end

@truncate_stacktrace LinearComplementarityProblem

const LCP = LinearComplementarityProblem

struct MixedLinearComplementarityProblem{MType, qType, uType, LB, UB} <:
       AbstractComplementarityProblem{false}
    M::MType
    q::qType
    u0::uType
    lb::LB
    ub::UB
end

@truncate_stacktrace MixedLinearComplementarityProblem

const MLCP = MixedLinearComplementarityProblem

function MLCP(prob::LCP)
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MLCP{false}(prob.M, prob.q, prob.u0, lb, ub)
end

function MLCP{iip}(M::MType,
    q::qType,
    u0::uType,
    lb::LB,
    ub::UB) where {iip, MType, qType, uType, LB, UB}
    return MLCP{iip, MType, qType, uType, LB, UB}(M, q, u0, lb, ub)
end

struct NonlinearComplementarityProblem{iip, F <: Function, uType, pType} <:
       AbstractComplementarityProblem{iip}
    f::F
    u0::uType
    p::pType
end

@truncate_stacktrace NonlinearComplementarityProblem 1

const NCP = NonlinearComplementarityProblem

function NCP(prob::LCP)
    return NCP{false}((x, θ) -> θ.M * x + θ.q, prob.u0, ComponentArray((; prob.M, prob.q)))
end

function NCP{iip}(f::F, u0::uType, p::pType=∅p) where {iip, F, uType, pType}
    return NCP{iip, F, uType, pType}(f, u0, p)
end

struct MixedComplementarityProblem{iip, F <: Function, uType, LB, UB, pType} <:
       AbstractComplementarityProblem{iip}
    f::F
    u0::uType
    lb::LB
    ub::UB
    p::pType
end

@truncate_stacktrace MixedComplementarityProblem 1

const MCP = MixedComplementarityProblem

MCP(prob::LCP) = MCP(NCP(prob))

function MCP(prob::MLCP{iip}) where {iip}
    p = ComponentArray((; prob.M, prob.q))
    return MCP{iip}((x, θ) -> θ.M * x + θ.q, prob.u0, prob.lb, prob.ub, p)
end

function MCP(prob::NCP{iip}) where {iip}
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MCP{iip}(prob.f, prob.u0, lb, ub, prob.p)
end

function MCP{iip}(f::F,
    u0::uType,
    lb::LB,
    ub::UB,
    p::pType=∅p) where {iip, F, uType, LB, UB, pType}
    return MCP{iip, F, uType, LB, UB, pType}(f, u0, lb, ub, p)
end

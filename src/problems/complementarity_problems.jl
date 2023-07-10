abstract type AbstractComplementarityProblem{iip} end

@concrete struct LinearComplementarityProblem <: AbstractComplementarityProblem{false}
    M
    q
    u0
end

@truncate_stacktrace LinearComplementarityProblem

const LCP = LinearComplementarityProblem

@concrete struct MixedLinearComplementarityProblem <: AbstractComplementarityProblem{false}
    M
    q
    u0
    lb
    ub
end

@truncate_stacktrace MixedLinearComplementarityProblem

const MLCP = MixedLinearComplementarityProblem

function MLCP(prob::LCP)
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MLCP(prob.M, prob.q, prob.u0, lb, ub)
end

@concrete struct NonlinearComplementarityProblem{iip, F <: Function} <:
                 AbstractComplementarityProblem{iip}
    f::F
    u0
    p
end

@truncate_stacktrace NonlinearComplementarityProblem 1

const NCP = NonlinearComplementarityProblem

function NCP(prob::LCP)
    return NCP{false}((x, θ) -> θ.M * x + θ.q, prob.u0, ComponentArray((; prob.M, prob.q)))
end

@concrete struct MixedComplementarityProblem{iip, F <: Function} <:
                 AbstractComplementarityProblem{iip}
    f::F
    u0
    lb
    ub
    p
end

@truncate_stacktrace MixedComplementarityProblem 1

const MCP = MixedComplementarityProblem

MCP(prob::LCP) = MCP(NCP(prob))

function MCP(prob::MLCP)
    p = ComponentArray((; prob.M, prob.q))
    return MCP{false}((x, θ) -> θ.M * x + θ.q, prob.u0, prob.lb, prob.ub, p)
end

function MCP(prob::NCP{iip}) where {iip}
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MCP{iip}(prob.f, prob.u0, lb, ub, prob.p)
end

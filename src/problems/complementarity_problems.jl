abstract type AbstractComplementarityProblem{iip, batched} end

@concrete struct LinearComplementarityProblem{iip, batched} <: AbstractComplementarityProblem{iip, batched}
    M
    q
    u0
    # function LinearComplementarityProblem{iip}(M, q, u0=nothing) where {iip}
    #     # By default, set iip to true since that is faster
    #     # For AD support, we need to set iip to false
    #     if u0 !== nothing && ndims(u0) == 2 && ndims(M) == 2 && ndims(q) == 1
    #         # If u0 is batched while problem is not, then reshape the problem
    #         M = repeat(reshape(M, size(M)..., 1); outer=(1, 1, size(u0, 2)))
    #         q = repeat(reshape(q, length(q), 1); outer=(1, size(u0, 2)))
    #     end
    #     batched = ndims(M) == 3
    #     if u0 === nothing
    #         u0 = zero(q)
    #     elseif batched && ndims(u0) == 1
    #         @warn "Incorrect batched version specification for `u0`. Reshaping to \
    #             ($(length(u0)), 1)."
    #         u0 = reshape(u0, :, 1)
    #     end
    #     if batched
    #         if !(ndims(q) == ndims(u0) == 2)
    #             throw(ArgumentError("Incorrect batched version specification: \
    #                 ndims(M) = 3, ndims(q) = $(ndims(q)), \
    #                 ndims(u0) = $(ndims(u0))"))
    #         end
    #         if size(u0, ndims(u0)) != size(q, ndims(q)) != size(M, ndims(M))
    #             throw(ArgumentError("Batch Sizes are inconsistent across M, q, u0"))
    #         end
    #     end
    #     return new{iip, batched, typeof(M), typeof(q), typeof(u0)}(M, q, u0)
    # end

    # LinearComplementarityProblem(args...) = LinearComplementarityProblem{true}(args...)
end

# function CRC.rrule(::Type{LinearComplementarityProblem}, args...; kwargs...)
#     function ∇LinearComplementarityProblem(Δ)
#         ∂M = __notangent(Δ) ? ∂∅ : Δ.M
#         ∂q = __notangent(Δ) ? ∂∅ : Δ.q
#         return ∂∅, ∂M, ∂q, ∂0
#     end
#     return LinearComplementarityProblem(args...; kwargs...), ∇LinearComplementarityProblem
# end

@truncate_stacktrace LinearComplementarityProblem

const LCP = LinearComplementarityProblem

@concrete struct MixedLinearComplementarityProblem <: AbstractComplementarityProblem{false, false}
    M
    q
    u0
    lb
    ub
end

@truncate_stacktrace MixedLinearComplementarityProblem

const MLCP = MixedLinearComplementarityProblem

# function MLCP(prob::LCP)
#     lb = zero(prob.u0)
#     ub = similar(prob.u0)
#     fill!(ub, eltype(prob.u0)(Inf))
#     return MLCP(prob.M, prob.q, prob.u0, lb, ub)
# end

# function MLCP{iip}(M::MType,
#     q::qType,
#     u0::uType,
#     lb::LB,
#     ub::UB) where {iip, MType, qType, uType, LB, UB}
#     return MLCP{iip, MType, qType, uType, LB, UB}(M, q, u0, lb, ub)
# end

@concrete struct NonlinearComplementarityProblem{iip, F <: Function} <:
    AbstractComplementarityProblem{iip, false}
    f::F
    u0
    p
end

@truncate_stacktrace NonlinearComplementarityProblem 1

const NCP = NonlinearComplementarityProblem

# function NCP(prob::LCP{iip, false}) where {iip}
#     θ = ComponentArray((; prob.M, prob.q))
#     if iip
#         function f!(y, x, θ)
#             y .= θ.q
#             mul!(y, θ.M, x, true, true)
#             return y
#         end
#         return NCP{true}(f!, prob.u0, θ)
#     else
#         return NCP{false}((x, θ) -> θ.M * x + θ.q, prob.u0, θ)
#     end
# end


function NCP{iip}(f::F, u0::uType, p::pType=∅p) where {iip, F, uType, pType}
    return NCP{iip, F, uType, pType}(f, u0, p)
end

@concrete struct MixedComplementarityProblem{iip, F <: Function} <:
    AbstractComplementarityProblem{iip, false}
    f::F
    u0
    lb
    ub
    p
end

@truncate_stacktrace MixedComplementarityProblem 1

const MCP = MixedComplementarityProblem

# MCP(prob::LCP) = MCP(NCP(prob))

# function MCP(prob::MLCP)
#     p = ComponentArray((; prob.M, prob.q))
#     return MCP{false}((x, θ) -> θ.M * x + θ.q, prob.u0, prob.lb, prob.ub, p)
# end

# function MCP(prob::NCP{iip}) where {iip}
#     lb = zero(prob.u0)
#     ub = similar(prob.u0)
#     fill!(ub, eltype(prob.u0)(Inf))
#     return MCP{iip}(prob.f, prob.u0, lb, ub, prob.p)
# end

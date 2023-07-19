abstract type AbstractComplementarityProblem{iip} end
abstract type AbstractLinearComplementarityProblem{iip, batched} <:
              AbstractComplementarityProblem{iip} end
abstract type AbstractNonlinearComplementarityProblem{iip} <:
              AbstractComplementarityProblem{iip} end

SciMLBase.isinplace(::AbstractComplementarityProblem{iip}) where {iip} = iip
function isbatched(::AbstractLinearComplementarityProblem{
    iip,
    batched,
}) where {iip, batched}
    return batched
end

@concrete struct LinearComplementarityProblem{iip, batched} <:
                 AbstractLinearComplementarityProblem{iip, batched}
    M
    q
    u0
end

function LinearComplementarityProblem{iip}(M, q, u0=nothing) where {iip}
    # By default, set iip to true since that is faster
    if u0 !== nothing && ndims(u0) == 2 && ndims(M) == 2 && ndims(q) == 1
        # If u0 is batched while problem is not, then reshape the problem
        M = reshape(M, size(M)..., 1)
        q = reshape(q, length(q), 1)
    end

    if ndims(M) == 3 && ndims(q) == 1
        q = reshape(q, length(q), 1)
    elseif ndims(M) == 2 && ndims(q) == 2
        M = reshape(M, size(M)..., 1)
    end

    batched = ndims(M) == 3
    batched && (batch_size = __check_correct_batching(M, q))

    if u0 === nothing
        u0 = similar(q, batched ? (size(q, 1), batch_size) : size(q))
        fill!(u0, 0)
    elseif batched
        @assert ndims(u0) == 2
        batch_size > 1 && @assert size(u0, 2) == batch_size
    end

    return LinearComplementarityProblem{iip, batched}(M, q, u0)
end

LinearComplementarityProblem(args...) = LinearComplementarityProblem{true}(args...)

for iip in (true, false)
    @eval function CRC.rrule(::Type{LinearComplementarityProblem{$iip}},
        M,
        q,
        args...;
        kwargs...)
        prob = LinearComplementarityProblem{$iip}(M, q, args...; kwargs...)
        function ∇LinearComplementarityProblem(Δ)
            if __notangent(Δ)
                ∂M = ∂∅
                ∂q = ∂∅
            else
                if isbatched(prob)
                    if ndims(M) != ndims(Δ.M)
                        ∂M = dropdims(sum(Δ.M; dims=ndims(Δ.M)); dims=ndims(Δ.M))
                    end
                    if ndims(q) != ndims(Δ.q)
                        ∂q = dropdims(sum(Δ.q; dims=ndims(Δ.q)); dims=ndims(Δ.q))
                    end
                end
                @isdefined(∂M) || (∂M = Δ.M)
                @isdefined(∂q) || (∂q = Δ.q)
            end
            return ∂∅, ∂M, ∂q, ∂0
        end
        return prob, ∇LinearComplementarityProblem
    end
end

@truncate_stacktrace LinearComplementarityProblem 1 2

const LCP = LinearComplementarityProblem

function (prob::LCP{iip, batched})(u0=prob.u0, M=prob.M, q=prob.q) where {iip, batched}
    f, u0 = if iip
        if batched
            function f_batched!(out, u, θ)
                M = reshape(view(θ, 1:length(M)), size(M))
                q = reshape(view(θ, (length(M) + 1):length(θ)), size(q, 1), 1, :)
                out .= q
                batched_mul!(out, M, reshape(u, size(u, 1), 1, :), true, true)
                return out
            end
            f_batched!, reshape(u0, size(u0, 1), 1, :)
        else
            function f_unbatched!(out, u, θ)
                M = reshape(view(θ, 1:length(M)), size(M))
                q = view(θ, (length(M) + 1):length(θ))
                out .= q
                mul!(out, M, u, true, true)
                return out
            end
            f_unbatched!, u0
        end
    else
        if batched
            function f_batched(u, θ)
                M = reshape(view(θ, 1:length(M)), size(M))
                q = reshape(view(θ, (length(M) + 1):length(θ)), size(q, 1), 1, :)
                return M ⊠ u .+ q
            end
            f_batched, reshape(u0, size(u0, 1), 1, :)
        else
            function f_unbatched(u, θ)
                M = reshape(view(θ, 1:length(M)), size(M))
                q = view(θ, (length(M) + 1):length(θ))
                return M * u .+ q
            end
            f_unbatched, u0
        end
    end

    θ = vcat(vec(M), vec(q))

    return f, u0, θ
end

@concrete struct MixedLinearComplementarityProblem{iip, batched} <:
                 AbstractLinearComplementarityProblem{iip, batched}
    M
    q
    u0
    lb
    ub
end

@truncate_stacktrace MixedLinearComplementarityProblem

const MLCP = MixedLinearComplementarityProblem

function MLCP(prob::LCP{iip, batched}) where {iip, batched}
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MLCP{iip, batched}(prob.M, prob.q, prob.u0, lb, ub)
end

@concrete struct NonlinearComplementarityProblem{iip, F <: Function} <:
                 AbstractNonlinearComplementarityProblem{iip}
    f::F
    u0
    p
end

@truncate_stacktrace NonlinearComplementarityProblem 1

const NCP = NonlinearComplementarityProblem

NCP(prob::LCP{iip}) where {iip} = NCP{iip}(prob()...)

@concrete struct MixedComplementarityProblem{iip, F <: Function} <:
                 AbstractNonlinearComplementarityProblem{iip}
    f::F
    u0
    lb
    ub
    p
end

@truncate_stacktrace MixedComplementarityProblem 1

const MCP = MixedComplementarityProblem

MCP(prob::LCP) = MCP(NCP(prob))

function MCP(prob::NCP{iip}) where {iip}
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MCP{iip}(prob.f, prob.u0, lb, ub, prob.p)
end

MCP(f, u0, lb, ub, p) = MCP{SciMLBase.isinplace(f, 3)}(f, u0, lb, ub, p)

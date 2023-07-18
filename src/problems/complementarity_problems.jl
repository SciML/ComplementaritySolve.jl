abstract type AbstractComplementarityProblem{iip, batched} end

SciMLBase.isinplace(::AbstractComplementarityProblem{iip}) where {iip} = iip
isbatched(::AbstractComplementarityProblem{iip, batched}) where {iip, batched} = batched

@concrete struct LinearComplementarityProblem{iip, batched} <:
                 AbstractComplementarityProblem{iip, batched}
    M
    q
    u0
end

function LinearComplementarityProblem{iip}(M, q, u0=nothing) where {iip}
    # By default, set iip to true since that is faster
    # For AD support, we need to set iip to false
    if u0 !== nothing && ndims(u0) == 2 && ndims(M) == 2 && ndims(q) == 1
        # If u0 is batched while problem is not, then reshape the problem
        M = repeat(reshape(M, size(M)..., 1); outer=(1, 1, size(u0, 2)))
        q = repeat(reshape(q, length(q), 1); outer=(1, size(u0, 2)))
    end
    batched = ndims(M) == 3
    if u0 === nothing
        u0 = zero(q)
    elseif batched && ndims(u0) == 1
        @warn "Incorrect batched version specification for `u0`. Reshaping to \
            ($(length(u0)), 1)."
        u0 = reshape(u0, :, 1)
    end
    if batched
        if !(ndims(q) == ndims(u0) == 2)
            throw(ArgumentError("Incorrect batched version specification: \
                ndims(M) = 3, ndims(q) = $(ndims(q)), \
                ndims(u0) = $(ndims(u0))!"))
        end
        if size(u0, ndims(u0)) != size(q, ndims(q)) ||
           size(q, ndims(q)) != size(M, ndims(M)) ||
           size(u0, ndims(u0)) != size(M, ndims(M))
            throw(ArgumentError("Batch Sizes are inconsistent across M, q, u0!"))
        end
    else
        if ndims(q) != 1 || ndims(u0) != 1
            throw(ArgumentError("`M` is not batched, but `q` ($(ndims(q))) or `u0` \
                                 ($(ndims(u0))) are!"))
        end
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
                if isbatched(prob) && ndims(M) != ndims(Δ.M) && ndims(q) != ndims(Δ.q)
                    ∂M = dropdims(sum(Δ.M; dims=ndims(Δ.M)); dims=ndims(Δ.M))
                    ∂q = dropdims(sum(Δ.q; dims=ndims(Δ.q)); dims=ndims(Δ.q))
                else
                    ∂M = Δ.M
                    ∂q = Δ.q
                end
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
                 AbstractComplementarityProblem{iip, batched}
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
                 AbstractComplementarityProblem{iip, false}
    f::F
    u0
    p
end

@truncate_stacktrace NonlinearComplementarityProblem 1

const NCP = NonlinearComplementarityProblem

function NCP(prob::LCP{iip, batched}) where {iip, batched}
    f, u0, θ = prob()
    return NCP{iip, batched}(f, u0, θ)
end

@concrete struct MixedComplementarityProblem{iip, batched, F <: Function} <:
                 AbstractComplementarityProblem{iip, batched}
    f::F
    u0
    lb
    ub
    p
end

@truncate_stacktrace MixedComplementarityProblem 1

const MCP = MixedComplementarityProblem

MCP(prob::LCP) = MCP(NCP(prob))

function MCP(prob::NCP{iip, batched}) where {iip, batched}
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MCP{iip, batched}(prob.f, prob.u0, lb, ub, prob.p)
end

MCP(f, u0, lb, ub, p) = MCP{SciMLBase.isinplace(f, 3)}(f, u0, lb, ub, p)

function MCP{iip}(f, u0, lb, ub, p) where {iip}
    batched = ndims(u0) ≥ 2 # Assume batched in this case
    return MCP{iip, batched}(f, u0, lb, ub, p)
end

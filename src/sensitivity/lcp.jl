@concrete struct LinearComplementarityAdjoint
    linsolve
end

LinearComplementarityAdjoint() = LinearComplementarityAdjoint(nothing)

@truncate_stacktrace LinearComplementarityAdjoint

_Jq(z) = __diagonal(one.(z))

__get_lcp_dimensions(z::AbstractVector, M) = (length(z), -1), length(z)^2
__get_lcp_dimensions(z::AbstractMatrix, M) = size(z), prod(size(M)[1:2])

function __lcp_gradient_computation(z::AbstractVector,
    ∂z,
    ∂ϕ₋∂u₋,
    M,
    ∂ϕ₋∂v₋,
    L,
    Lₘ,
    _,
    linsolve)
    A = M' * ∂ϕ₋∂u₋ + ∂ϕ₋∂v₋
    # Following line is same as -∂ϕ₋∂u₋ * reduce(hcat, Zygote.jacobian((A, q) -> A * z .+ q, A, q))
    B = -hcat(reshape(reshape(z, 1, 1, L) .* reshape(∂ϕ₋∂u₋, L, L, 1), L, Lₘ),
        ∂ϕ₋∂u₋ * _Jq(z))
    λ = solve(LinearProblem(A, __unfillarray(∂z)), linsolve).u
    return vec(λ' * B)
end

function __lcp_gradient_computation(z::AbstractMatrix,
    ∂z,
    ∂ϕ₋∂u₋,
    M,
    ∂ϕ₋∂v₋,
    L,
    Lₘ,
    N,
    linsolve)
    A = __make_block_diagonal_operator(batched_transpose(M) ⊠ ∂ϕ₋∂u₋ .+ ∂ϕ₋∂v₋)
    B = -hcat(reshape(reshape(z, 1, 1, L, N) .* reshape(∂ϕ₋∂u₋, L, L, 1, N), L, Lₘ, N),
        ∂ϕ₋∂u₋ ⊠ _Jq(z))
    λ = reshape(solve(LinearProblem(A, __unfillarray(vec(∂z))), linsolve).u, L, N)
    return dropdims(reshape(λ, 1, L, N) ⊠ B; dims=1)
end

@views function ∇linear_complementarity_problem!(alg::LinearComplementarityAdjoint,
    ::Val{iip},
    ::Val{batched},
    ∂z,
    z,
    ∂M,
    M,
    ∂q,
    q) where {iip, batched}
    ∂z === nothing && return

    (L, N), Lₘ = __get_lcp_dimensions(z, M)

    u₋ = batched ? (batched_mul(M, reshape(z, size(z, 1), 1, :))[:, 1, :] .+ q) :
         (M * z + q)
    v₋ = z

    den = @. inv(√(u₋^2 + v₋^2))
    ∂ϕ₋∂u₋ = __diagonal(@. 1 - u₋ * den)
    ∂ϕ₋∂v₋ = __diagonal(@. 1 - v₋ * den)

    ∂Mq = __lcp_gradient_computation(z, ∂z, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, N, alg.linsolve)

    vec(∂M) .+= vec(selectdim(∂Mq, 1, 1:Lₘ))
    vec(∂q) .+= vec(selectdim(∂Mq, 1, (Lₘ + 1):size(∂Mq, 1)))

    return
end

# FIXME: All calls to solve now go through this function, but it should be dispatched only
#        if the sensealg is LinearComplementarityAdjoint.
function CRC.rrule(::typeof(solve),
    prob::LinearComplementarityProblem{iip, batched},
    alg;
    sensealg=LinearComplementarityAdjoint(),
    kwargs...) where {iip, batched}
    sol = solve(prob, alg; kwargs...)

    function ∇lcpsolve(Δ)
        ∂M, ∂q = zero(prob.M), zero(prob.q)
        ∇linear_complementarity_problem!(sensealg,
            Val(iip),
            Val(batched),
            __nothingify(Δ.u),
            sol.u,
            ∂M,
            prob.M,
            ∂q,
            prob.q)
        ∂prob = (; M=∂M, q=∂q, u0=∂∅)
        return ∂∅, ∂prob, ∂∅
    end

    return sol, ∇lcpsolve
end

@concrete struct LinearComplementarityAdjoint <: AbstractComplementaritySensitivityAlgorithm
    linsolve
end

LinearComplementarityAdjoint() = LinearComplementarityAdjoint(nothing)

@truncate_stacktrace LinearComplementarityAdjoint

__lcp_dims(u::AbstractVector, M) = (length(u), -1), length(u)^2
__lcp_dims(u::AbstractMatrix, M) = size(u), prod(size(M)[1:2])

function __∇lcp(u::AbstractVector, ∂u, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, linsolve)
    A = M' * ∂ϕ₋∂u₋ + ∂ϕ₋∂v₋
    # Following line is same as
    # -∂ϕ₋∂u₋ * reduce(hcat, Zygote.jacobian((A, q) -> A * u .+ q, A, q))
    B = -hcat(reshape(reshape(u, 1, 1, L) .* reshape(∂ϕ₋∂u₋, L, L, 1), L, Lₘ),
        ∂ϕ₋∂u₋ * __diagonal(one.(u)))
    λ = solve(LinearProblem(A, __unfillarray(∂u)), linsolve).u
    return vec(λ' * B)
end

function __∇lcp(u::AbstractMatrix, ∂u, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, linsolve)
    A = __make_block_diagonal_operator(batched_transpose(M) ⊠ ∂ϕ₋∂u₋ .+ ∂ϕ₋∂v₋)
    B = -hcat(reshape(reshape(u, 1, 1, L, :) .* reshape(∂ϕ₋∂u₋, L, L, 1, :), L, Lₘ, :),
        ∂ϕ₋∂u₋ ⊠ __diagonal(one.(u)))
    λ = reshape(solve(LinearProblem(A, __unfillarray(vec(∂u))), linsolve).u, L, :)
    return dropdims(reshape(λ, 1, L, :) ⊠ B; dims=1)
end

@views function __solve_adjoint(prob::LinearComplementarityProblem{iip, batched},
    sensealg::LinearComplementarityAdjoint,
    sol,
    ∂sol,
    u0,
    M,
    q;
    kwargs...) where {iip, batched}
    (__notangent(∂sol) || __notangent(∂sol.u)) && return (∂∅, ∂∅)

    u, ∂u = sol.u, ∂sol.u
    ∂M, ∂q = zero(M), zero(q)

    (L, N), Lₘ = __lcp_dims(u, M)

    u₋ = batched ? (batched_mul(M, reshape(u, size(u, 1), 1, :))[:, 1, :] .+ q) :
         (M * u .+ q)
    v₋ = u

    den = @. inv(√(u₋^2 + v₋^2))
    ∂ϕ₋∂u₋ = __diagonal(@. 1 - u₋ * den)
    ∂ϕ₋∂v₋ = __diagonal(@. 1 - v₋ * den)

    ∂Mq = __∇lcp(u, ∂u, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, sensealg.linsolve)

    vec(∂M) .+= vec(selectdim(∂Mq, 1, 1:Lₘ))
    vec(∂q) .+= vec(selectdim(∂Mq, 1, (Lₘ + 1):size(∂Mq, 1)))

    return (∂M, ∂q)
end

@concrete struct LinearComplementarityAdjoint <: AbstractComplementaritySensitivityAlgorithm
    linsolve
end

LinearComplementarityAdjoint() = LinearComplementarityAdjoint(nothing)

@truncate_stacktrace LinearComplementarityAdjoint

__lcp_dims(u::AV, M) = (length(u), -1), length(u)^2
__lcp_dims(u::AM, M) = size(u), prod(size(M)[1:2])

function __∇lcp(u::AV, ∂u, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, linsolve)
    A = M' * ∂ϕ₋∂u₋ + ∂ϕ₋∂v₋
    # Following line is same as
    # -∂ϕ₋∂u₋ * reduce(hcat, Zygote.jacobian((A, q) -> A * u .+ q, A, q))
    B = -hcat(reshape(reshape(u, 1, 1, L) .* reshape(∂ϕ₋∂u₋, L, L, 1), L, Lₘ),
        ∂ϕ₋∂u₋ * __diagonal(one.(u)))
    if linsolve === nothing
        # FIXME: Default linsolve selection in LinearSolve.jl fails on GPU
        λ = A \ __unfillarray(∂u)
    else
        λ = solve(LinearProblem(A, __unfillarray(∂u)), linsolve).u
    end
    return vec(λ' * B)
end

function __∇lcp(u::AM, ∂u, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, linsolve)
    A = __make_block_diagonal_operator(batched_transpose(M) ⊠ ∂ϕ₋∂u₋ .+ ∂ϕ₋∂v₋)
    B = -hcat(reshape(reshape(u, 1, 1, L, :) .* reshape(∂ϕ₋∂u₋, L, L, 1, :), L, Lₘ, :),
        ∂ϕ₋∂u₋ ⊠ __diagonal(one.(u)))
    λ = solve(LinearProblem(A, __unfillarray(vec(∂u))), linsolve).u
    return dropdims(reshape(λ, 1, L, :) ⊠ B; dims=1)
end

@views function __solve_adjoint(prob::LinearComplementarityProblem,
    sensealg::LinearComplementarityAdjoint, sol, ∂sol, u0, M, q; kwargs...)
    (__notangent(∂sol) || __notangent(∂sol.u)) && return (∂∅, ∂∅)

    u, ∂u = sol.u, ∂sol.u

    (L, _), Lₘ = __lcp_dims(u, M)

    u₋ = matmul(M, u) .+ q
    v₋ = u

    den = @. inv(√(u₋^2 + v₋^2))
    ∂ϕ₋∂u₋ = __diagonal(@. 1 - u₋ * den)
    ∂ϕ₋∂v₋ = __diagonal(@. 1 - v₋ * den)

    ∂Mq = __∇lcp(u, ∂u, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, sensealg.linsolve)
    ∂M_ = selectdim(∂Mq, 1, 1:Lₘ)
    ∂q_ = selectdim(∂Mq, 1, (Lₘ + 1):size(∂Mq, 1))

    if isbatched(prob)
        size(∂M_, 2) != size(M, 3) && (∂M_ = sum(∂M_; dims=2))
        size(∂q_, 2) != size(q, 2) && (∂q_ = sum(∂q_; dims=2))
    end

    return (reshape(∂M_, size(M)), reshape(∂q_, size(q)))
end

# Linear Complementarity Problem
@kwdef struct LinearComplementarityAdjoint{L}
    linsolve::L = nothing
end

@truncate_stacktrace LinearComplementarityAdjoint

_Jq(z) = __diagonal((x -> isapprox(x, 0; rtol=1e-5, atol=1e-5) ? x : one(x)).(z))

__get_lcp_dimensions(z::AbstractVector) = (length(z), -1), length(z)^2
__get_lcp_dimensions(z::AbstractMatrix) = (size(z, 1), size(z, 2)), size(z, 1)^2

function __lcp_gradient_computation(z::AbstractVector,
    ∂z,
    ∂ϕ₋∂u₋,
    M,
    ∂ϕ₋∂v₋,
    L,
    Lₘ,
    _,
    linsolve)
    A = ∂ϕ₋∂u₋ * M + ∂ϕ₋∂v₋
    B = -hcat(reshape(reshape(z, 1, 1, L) .* reshape(∂ϕ₋∂u₋, L, L, 1), L, Lₘ), _Jq(z))
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
    A = __make_block_diagonal_matrix(∂ϕ₋∂u₋ ⊠ reshape(M, L, L, 1) .+ ∂ϕ₋∂v₋)
    B = -hcat(reshape(reshape(z, 1, 1, L, N) .* reshape(∂ϕ₋∂u₋, L, L, 1, N), L, Lₘ, N),
        _Jq(z))
    λ = reshape(solve(LinearProblem(A, __unfillarray(vec(∂z))), linsolve).u, L, N)
    return vec(sum(reshape(λ, 1, L, N) ⊠ B; dims=3))
end

@views function ∇linear_complementarity_problem!(alg::LinearComplementarityAdjoint,
    ∂z,
    z,
    ∂w,
    w,
    ∂M,
    M,
    ∂q,
    q)
    ∂w === nothing && ∂z === nothing && return

    if ∂w !== nothing
        sum!(∂q, ∂w)
        mul!(∂M, ∂w, z')
        if ∂z === nothing
            ∂z = M' * ∂w
        elseif ArrayInterfaceCore.can_setindex(∂z)
            ∂z .+= M' * ∂w
        else
            ∂z = ∂z .+ M' * ∂w
        end
    end

    (L, N), Lₘ = __get_lcp_dimensions(z)

    u₋, v₋ = w, z
    den = @. inv(√(u₋^2 + v₋^2))
    ∂ϕ₋∂u₋ = __diagonal(@. 1 - u₋ * den)
    ∂ϕ₋∂v₋ = __diagonal(@. 1 - v₋ * den)

    ∂Mq = __lcp_gradient_computation(z, ∂z, ∂ϕ₋∂u₋, M, ∂ϕ₋∂v₋, L, Lₘ, N, alg.linsolve)

    vec(∂M) .+= vec(∂Mq[1:Lₘ])
    vec(∂q) .+= vec(∂Mq[(Lₘ + 1):end])

    return
end

function CRC.rrule(::typeof(solve),
    prob::LinearComplementarityProblem,
    alg;
    sensealg=LinearComplementarityAdjoint(),
    kwargs...)
    sol = solve(prob, alg; kwargs...)

    function ∇lcpsolve(Δ)
        ∂M, ∂q = zero(prob.M), zero(prob.q)
        ∇linear_complementarity_problem!(sensealg,
            __nothingify(Δ.z),
            sol.z,
            __nothingify(Δ.w),
            sol.w,
            ∂M,
            prob.M,
            ∂q,
            prob.q)
        ∂prob = (; M=∂M, q=∂q, u0=∂∅)
        return ∂∅, ∂prob, ∂∅
    end

    return sol, ∇lcpsolve
end

# Mixed Complementarity Problem
@kwdef struct MixedComplementarityAdjoint{L}
    linsolve::L = nothing
end

@truncate_stacktrace MixedComplementarityAdjoint

@views function ∇mixed_complementarity_problem!(cfg::RuleConfig{>:HasReverseMode},
    alg::MixedComplementarityAdjoint,
    ∂u,
    u,
    ∂p,
    p,
    f,
    lb,
    ub)
    ∂u === nothing && return

    fᵤ = f(u, p)
    ∂ϕ₊∂u₊, ∂ϕ₊∂v₊ = Jϕ₊(fᵤ, u, ub)
    ∂ϕ₋∂u₋, ∂ϕ₋∂v₋ = Jϕ₋(fᵤ, u, lb)

    A₁ = ∂ϕ₊∂u₊ * ∂ϕ₋∂u₋
    A₂ = ∂ϕ₊∂v₊ * ∂ϕ₋∂u₋ + ∂ϕ₋∂v₋
    if length(u) ≤ 50
        # Construct the Full Matrix
        A = only(Zygote.jacobian(x -> f(x, p), u))' * A₁ .+ A₂
    else
        # Use Matrix Free Methods
        A = fixed_vecjac_operator(f, u, p, A₁, A₂)
    end
    λ = solve(LinearProblem(A, __unfillarray(∂u)), alg.linsolve).u

    _, pb_f = Zygote.pullback(p -> f(u, p), p)
    vec(∂p) .= -vec(only(pb_f((A₁ * λ)')))

    return
end

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode},
    ::typeof(solve),
    prob::MixedComplementarityProblem{false},
    alg;
    sensealg=MixedComplementarityAdjoint(),
    kwargs...)
    sol = solve(prob, alg; kwargs...)

    function ∇mcpsolve(Δ)
        ∂p = zero(prob.p)
        ∇mixed_complementarity_problem!(cfg,
            sensealg,
            __nothingify(Δ.u),
            sol.u,
            ∂p,
            prob.p,
            prob.f,
            prob.lb,
            prob.ub)
        ∂prob = (; p=∂p, u0=∂∅, lb=∂∅, ub=∂∅, f=∂∅)
        return ∂∅, ∂prob, ∂∅
    end

    return sol, ∇mcpsolve
end

function fixed_vecjac_operator(f, y, p, A₁, A₂)
    input, pb_f = Zygote.pullback(x -> f(x, p), y)
    output = only(pb_f(input))
    function f_operator!(du, u, p, t)
        λ = reshape(u, size(input))
        du .= vec(only(pb_f(A₁ * λ)) .+ A₂ * λ)
        return du
    end
    return FunctionOperator(f_operator!, vec(input), vec(output))
end

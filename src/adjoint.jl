# Linear Complementarity Problem
@kwdef struct LinearComplementarityAdjoint{L}
    linsolve::L = nothing
end

_Jq(z) = Diagonal((x -> isapprox(x, 0; rtol=1e-5, atol=1e-5) ? x : one(x)).(z))

@views function ∇linear_complementarity_problem!(alg::LinearComplementarityAdjoint, ∂z, z,
                                                 ∂w, w, ∂M, M, ∂q, q)
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

    u₋, v₋ = w, z
    den = @. inv(√(u₋^2 + v₋^2))
    ∂ϕ₋∂u₋ = Diagonal(@. 1 - u₋ * den)
    ∂ϕ₋∂v₋ = Diagonal(@. 1 - v₋ * den)

    A = ∂ϕ₋∂u₋ * M + ∂ϕ₋∂v₋
    B = -hcat(reshape(reshape(z, 1, 1, :) .* repeat(∂ϕ₋∂u₋; outer=(1, 1, length(z))),
                      length(z), length(M)), _Jq(z))

    λ = solve(LinearProblem(A, ∂z), alg.linsolve).u
    ∂Mq = λ' * B

    vec(∂M) .+= vec(∂Mq[1, 1:length(M)])
    vec(∂q) .+= vec(∂Mq[1, (length(M) + 1):end])

    return
end

function CRC.rrule(::typeof(solve), prob::LinearComplementarityProblem, alg;
                   sensealg=LinearComplementarityAdjoint(), kwargs...)
    sol = solve(prob, alg; kwargs...)

    function ∇lcpsolve(Δ)
        ∂M, ∂q = zero(prob.M), zero(prob.q)
        ∇linear_complementarity_problem!(sensealg, __nothingify(Δ.z), sol.z,
                                         __nothingify(Δ.w), sol.w, ∂M, prob.M, ∂q, prob.q)
        ∂prob = (; M=∂M, q=∂q, u0=∂∅)
        return ∂∅, ∂prob, ∂∅
    end

    return sol, ∇lcpsolve
end

# Mixed Complementarity Problem
@kwdef struct MixedComplementarityAdjoint{L}
    linsolve::L = nothing
end

@views function ∇mixed_complementarity_problem!(cfg::RuleConfig{>:HasReverseMode},
                                                alg::MixedComplementarityAdjoint, ∂u, u, ∂p,
                                                p, f, lb, ub)
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

function CRC.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(solve),
                   prob::MixedComplementarityProblem{false}, alg;
                   sensealg=MixedComplementarityAdjoint(), kwargs...)
    sol = solve(prob, alg; kwargs...)

    function ∇mcpsolve(Δ)
        ∂p = zero(prob.p)
        ∇mixed_complementarity_problem!(cfg, sensealg, __nothingify(Δ.u), sol.u, ∂p, prob.p,
                                        prob.f, prob.lb, prob.ub)
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

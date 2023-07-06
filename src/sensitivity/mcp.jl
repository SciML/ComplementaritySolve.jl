## FIXME: Support batching for the solvers
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
        A = __fixed_vecjac_operator(f, u, p, A₁, A₂)
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

## TODO: Use SparseDiffTools v2
function __fixed_vecjac_operator(f, y, p, A₁, A₂)
    input, pb_f = Zygote.pullback(x -> f(x, p), y)
    output = only(pb_f(input))
    function f_operator!(du, u, p, t)
        λ = reshape(u, size(input))
        du .= vec(only(pb_f(A₁ * λ)) .+ A₂ * λ)
        return du
    end
    return FunctionOperator(f_operator!, vec(input), vec(output))
end

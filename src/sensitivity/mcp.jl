@concrete struct MixedComplementarityAdjoint <: AbstractComplementaritySensitivityAlgorithm
    linsolve
end

MixedComplementarityAdjoint() = MixedComplementarityAdjoint(nothing)

@truncate_stacktrace MixedComplementarityAdjoint

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

@views function __solve_adjoint(prob::MixedComplementarityProblem{false},
    sensealg::MixedComplementarityAdjoint,
    sol,
    ∂sol,
    u0,
    p;
    kwargs...)
    (__notangent(∂sol) || __notangent(∂sol.u)) && return (∂∅,)

    (; f, lb, ub) = prob
    u, ∂u = sol.u, ∂sol.u

    fᵤ = f(u, p)
    ∂ϕ₊∂u₊, ∂ϕ₊∂v₊ = Jϕ₊(fᵤ, u, ub)
    ∂ϕ₋∂u₋, ∂ϕ₋∂v₋ = Jϕ₋(fᵤ, u, lb)

    A₁ = ∂ϕ₊∂u₊ * ∂ϕ₋∂u₋
    A₂ = ∂ϕ₊∂v₊ * ∂ϕ₋∂u₋ + ∂ϕ₋∂v₋
    if length(u) ≤ 50
        # Construct the Full Matrix
        A = only(Zygote.jacobian(Base.Fix2(f, p), u))' * A₁ .+ A₂
    else
        # Use Matrix Free Methods
        ## NOTE: If we use SparseDiffTools here we will have to mess around with a wrapper
        ##       over the FunctionOperator
        A = __fixed_vecjac_operator(f, u, p, A₁, A₂)
    end
    λ = solve(LinearProblem(A, __unfillarray(∂u)), sensealg.linsolve).u

    _, pb_f = Zygote.pullback(Base.Fix1(f, u), p)
    ∂p = -reshape(vec(only(pb_f((A₁ * λ)'))), size(p))

    return (∂p,)
end

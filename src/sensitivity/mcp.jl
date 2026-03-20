@concrete struct MixedComplementarityAdjoint <: AbstractComplementaritySensitivityAlgorithm
    linsolve
end

MixedComplementarityAdjoint() = MixedComplementarityAdjoint(nothing)

@truncate_stacktrace MixedComplementarityAdjoint

function __fixed_vecjac_operator(f, y, p, A₁, A₂)
    input, pb_f = Zygote.pullback(x -> f(x, p), y)
    output = only(pb_f(input))
    function f_operator!(du, u, _u, p, t)
        λ = reshape(u, size(input))
        du .= vec(only(pb_f(A₁ * λ)) .+ A₂ * λ)
        return du
    end
    return FunctionOperator(f_operator!, vec(input), vec(output); isinplace = true)
end

@views function __solve_adjoint(
        prob::MixedComplementarityProblem,
        sensealg::MixedComplementarityAdjoint, sol, ∂sol, u0, p; kwargs...
    )
    (__notangent(∂sol) || __notangent(∂sol.u)) && return (∂∅,)

    (; f, lb, ub) = prob
    u, ∂u = sol.u, ∂sol.u

    if isinplace(prob)
        fᵤ = similar(u)
        f(fᵤ, u, p)
    else
        fᵤ = f(u, p)
    end
    ∂ϕ₊∂u₊, ∂ϕ₊∂v₊ = Jϕ₊(fᵤ, u, ub)
    ∂ϕ₋∂u₋, ∂ϕ₋∂v₋ = Jϕ₋(fᵤ, u, lb)

    A₁ = ∂ϕ₊∂u₊ * ∂ϕ₋∂u₋
    A₂ = ∂ϕ₊∂v₊ * ∂ϕ₋∂u₋ + ∂ϕ₋∂v₋
    if isinplace(prob)
        # Using ForwardDiff for now. We can potentially use Enzyme.jl here
        J = ForwardDiff.jacobian((y, u) -> f(y, u, p), fᵤ, u)
        A = J' * A₁ .+ A₂
    else
        if length(u) ≤ 50
            # Construct the Full Matrix
            A = only(Zygote.jacobian(Base.Fix2(f, p), u))' * A₁ .+ A₂
        else
            # Use Matrix Free Methods
            ## NOTE: If we use SparseDiffTools here we will have to mess around with a wrapper
            ##       over the FunctionOperator
            A = __fixed_vecjac_operator(f, u, p, A₁, A₂)
        end
    end

    if sensealg.linsolve === nothing
        # FIXME: Default linsolve selection in LinearSolve.jl fails on GPU
        λ = A \ __unfillarray(∂u)
    else
        λ = solve(LinearProblem(A, __unfillarray(∂u)), sensealg.linsolve).u
    end

    if isinplace(prob)
        # Using ForwardDiff for now. We can potentially use Enzyme.jl here
        J = ForwardDiff.jacobian((y, p) -> f(y, u, p), fᵤ, p)
        ∂p = -reshape((A₁ * λ)' * J, size(p))
    else
        _, pb_f = Zygote.pullback(Base.Fix1(f, u), p)
        ∂p = -reshape(vec(only(pb_f((A₁ * λ)'))), size(p))
    end

    return (∂p,)
end

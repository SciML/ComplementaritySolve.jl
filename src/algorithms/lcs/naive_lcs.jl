@concrete struct NaiveLCSAlgorithm <: AbstractComplementaritySystemAlgorithm
    ode_solver
    lcp_solver
end

function solve(prob::LinearComplementaritySystem{sstate},
    alg::NaiveLCSAlgorithm;
    ode_kwargs=(;),
    lcp_kwargs=(;),
    kwargs...) where {sstate}
    (; A, B, D, a, F, E, c, x0, controller, λ0, p, tspan) = prob

    function dxdt(x, p, t)
        lcp_sol = solve(LCP(F, E * x .+ c, λ0), alg.lcp_solver; lcp_kwargs..., kwargs...)
        λ = lcp_sol.z
        u_ = controller(x, λ, p, t)
        return A * x .+ B * u_ .+ D * λ .+ a
    end

    ode_prob = ODEProblem(ODEFunction{false}(dxdt), x0, tspan, p)
    if sstate
        # Solve for Equilibrium instead of integrating till Inf.
        # Expect the ode_solver to be DynamicSS
        return solve(SteadyStateProblem(ode_prob), alg.ode_solver; ode_kwargs..., kwargs...)
    else
        return solve(ode_prob, alg.ode_solver; ode_kwargs..., kwargs...)
    end
end

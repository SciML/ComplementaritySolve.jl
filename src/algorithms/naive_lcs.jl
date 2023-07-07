struct NaiveLCSAlgorithm{O, L} <: AbstractComplementaritySystemAlgorithm
    ode_solver::O
    lcp_solver::L
end

function solve(prob::LinearComplementaritySystem,
    alg::NaiveLCSAlgorithm;
    ode_kwargs=(;),
    lcp_kwargs=(;),
    kwargs...)
    (; A, B, D, a, F, E, c, x0, u, λ0, p, tspan) = prob
    λ = λ0

    function dxdt(x, p, t)
        # QUESTION: Should be use λ or λ0?
        lcp_sol = solve(LCP(F, E * x .+ c, λ), alg.lcp_solver; lcp_kwargs..., kwargs...)
        λ = lcp_sol.z
        u_ = u(x, λ, p, t)
        return A * x .+ B * u_ .+ D * λ .+ a
    end

    ode_prob = ODEProblem(dxdt, x0, tspan, p)
    return solve(ode_prob, alg.ode_solver; ode_kwargs..., kwargs...)
end

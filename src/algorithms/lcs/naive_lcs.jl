"""
    NaiveLCSAlgorithm(ode_solver, lcp_solver)

Solve a [`LinearComplementaritySystem`](@ref) by solving an LCP inside each ODE
right-hand-side evaluation.

# Arguments

- `ode_solver`: OrdinaryDiffEq/SciML ODE solver used for the state dynamics, or a
    steady-state solver when the system has an infinite final time.
- `lcp_solver`: Complementarity algorithm used for each embedded LCP.

# Fields

- `ode_solver`: Solver for the continuous state dynamics.
- `lcp_solver`: Solver for the complementarity subproblem.

# Keywords Passed To `solve`

- `ode_kwargs`: Named tuple forwarded to the ODE or steady-state solve.
- `lcp_kwargs`: Named tuple forwarded to each embedded LCP solve.
- `kwargs...`: Additional keywords forwarded to both solve calls where applicable.
"""
@concrete struct NaiveLCSAlgorithm <: AbstractComplementaritySystemAlgorithm
    ode_solver
    lcp_solver
end

function solve(
        prob::LinearComplementaritySystem{sstate}, alg::NaiveLCSAlgorithm;
        ode_kwargs = (;), lcp_kwargs = (;), kwargs...
    ) where {sstate}
    (; A, B, D, a, F, E, c, x0, controller, λ0, p, tspan) = prob

    function dxdt(x, p, t)
        lcp_sol = solve(LCP(F, E * x .+ c, λ0), alg.lcp_solver; lcp_kwargs..., kwargs...)
        λ = lcp_sol.u
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

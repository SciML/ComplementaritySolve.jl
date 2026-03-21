using ChainRulesCore, ComplementaritySolve, ComponentArrays, DiffEqBase, ForwardDiff,
    LinearAlgebra, Optimization, OptimizationOptimisers, OrdinaryDiffEq, SciMLSensitivity,
    SimpleNonlinearSolve, SparseArrays, StableRNGs, Statistics, SteadyStateDiffEq, Test
using Zygote

const m₁ = 0.5f0
const m₂ = 1.0f0
const l₁ = 0.5f0
const l₂ = 1.0f0
const g = 9.81f0
const k = 1.0f0
const d = 1.0f0

const 𝒶 = m₁ * l₁^2 + m₂ * l₂^2
const 𝒷 = m₂ * l₂^2
const 𝒸 = m₂ * l₁ * l₂

const M = [𝒶 + 𝒷 + 2𝒸 𝒷 + 𝒸; 𝒷 + 𝒸 𝒷]
const Jᵀ = Float32[-1 1; 0 0]
const A = Float32[
    0 0 1 0;
    0 0 0 1;
    g / l₁ -(g * m₂) / (l₁ * m₁) 0 0;
    -g / l₁ (g * (l₁ * m₁ + l₁ * m₂ + l₂ * m₂)) / (l₁ * l₂ * m₁) 0 0
]
const B = reshape(
    Float32[
        0
        0
        -(l₁ + l₂) / (l₂ * m₁ * l₁^2)
        (m₁ * l₁^2 + m₂ * (l₁ + l₂)^2) / (m₁ * m₂ * l₁^2 * l₂^2)
    ],
    (4, 1)
)
const D = vcat(zeros(Float32, 2, 2), inv(M) * Jᵀ)
const a = 0.0f0

const E = sparse([1, 2], [1, 1], Float32[-1, 1], 2, 4)
const F = diagm(Float32[1 / k, 1 / k])
const c = d

rng = StableRNG(0)

x0 = vcat(
    randn(rng, Float32, 2) .* 0.005f0, randn(rng, Float32, 1) * 0.02f0,
    randn(rng, Float32, 1) * 0.01f0
)
tspan = (0.0f0, 1.0f0)

controller(x, λ, p, _) = p.K * x .+ p.L * λ

# Taken from https://arxiv.org/pdf/2008.02104.pdf Section 6.C
stable_K = Float32[73.07 38.11 30.41 18.95]
stable_L = Float32[-4.13 4.13]

stable_θ = ComponentArray(; K = stable_K, L = stable_L)

function finite_horizon_ode_test(θ)
    prob = LCS(x0, controller, tspan, θ, A, B, D, a, E, F, c)
    solver = NaiveLCSAlgorithm(Tsit5(), NonlinearReformulation())
    sol = solve(prob, solver; abstol = 1.0f-6, reltol = 1.0f-6)

    @test sol isa SciMLBase.ODESolution
    @test SciMLBase.successful_retcode(sol)
    @test all(Base.Fix1(all, isfinite), sol.u)

    ∂θ = only(
        Zygote.gradient(θ) do θ
            prob = LCS(x0, controller, tspan, θ, A, B, D, a, E, F, c)
            sol = solve(
                prob, solver;
                ode_kwargs = (; sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP())),
                lcp_kwargs = (; sensealg = LinearComplementarityAdjoint())
            )
            return sum(abs2, last(sol.u))
        end
    )

    @test all(isfinite, ∂θ)

    ∂θ_fd = ForwardDiff.gradient(θ) do θ
        prob = LCS(x0, controller, tspan, θ, A, B, D, a, E, F, c)
        sol = solve(prob, solver)
        return sum(abs2, last(sol.u))
    end

    return @test ∂θ ≈ ∂θ_fd atol = 1.0e-2 rtol = 1.0e-2
end

function steady_state_test(θ)
    prob = LCS(x0, controller, (first(tspan), Inf32), θ, A, B, D, a, E, F, c)
    solver = NaiveLCSAlgorithm(
        DynamicSS(Tsit5()),
        NonlinearReformulation()
    )
    sol = solve(prob, solver; abstol = 1.0f-2, reltol = 1.0f-2)

    @test sol isa SciMLBase.NonlinearSolution
    @test_broken SciMLBase.successful_retcode(sol)
    @test all(Base.Fix1(all, isfinite), sol.u)

    ∂θ = only(
        Zygote.gradient(θ) do θ
            prob = LCS(x0, controller, (first(tspan), Inf32), θ, A, B, D, a, E, F, c)
            sol = solve(
                prob, solver;
                ode_kwargs = (; sensealg = SteadyStateAdjoint(; autojacvec = ZygoteVJP())),
                lcp_kwargs = (; sensealg = LinearComplementarityAdjoint())
            )
            return sum(abs2, last(sol.u))
        end
    )

    @test all(isfinite, ∂θ)

    ∂θ_fd = ForwardDiff.gradient(θ) do θ
        prob = LCS(x0, controller, (first(tspan), Inf32), θ, A, B, D, a, E, F, c)
        sol = solve(prob, solver)
        return sum(abs2, last(sol.u))
    end

    return @test ∂θ ≈ ∂θ_fd atol = 1.0e-2 rtol = 1.0e-2
end

@testset "Stable Controller" begin
    @testset "Finite Horizon ODE" begin
        finite_horizon_ode_test(stable_θ)
    end

    @testset "Solve to Infinity (Steady-State)" begin
        steady_state_test(stable_θ)
    end
end

@testset "Parameter Estimation" begin
    prob = LCS(x0, controller, tspan, stable_θ, A, B, D, a, E, F, c)
    solver = NaiveLCSAlgorithm(Vern9(), NonlinearReformulation())
    sol = solve(prob, solver; abstol = 1.0f-8, reltol = 1.0f-8)

    @testset "# of Samples: $N" for N in (10, 100, 1000)
        N > length(sol.t) && continue
        # Generate some data
        idxs = unique(sort(rand(rng, 1:length(sol.t), N)))
        target_us = hcat(sol.u[idxs]...)
        ts = sol.t[idxs]

        # Initialize the controller
        θ_init = ComponentArray(;
            K = randn(rng, Float32, 1, 4),
            L = randn(rng, Float32, 1, 2)
        ) .* 0.0001f0
        solver = NaiveLCSAlgorithm(Tsit5(), NonlinearReformulation())

        function loss_function(θ)
            prob = LCS(x0, controller, tspan, θ, A, B, D, a, E, F, c)
            lcs_sol = solve(
                prob, solver; abstol = 1.0f-3, reltol = 1.0f-3,
                ode_kwargs = (; saveat = ts)
            )
            return mean(abs2, reduce(hcat, lcs_sol.u) .- target_us)
        end

        iter = 0

        function callback_parameter_estim(θ, loss)
            iter += 1
            if iter % 100 == 1 || loss ≤ 0.01f0
                @info "Parameter Estimation with $N datapoints" iter = iter loss = loss
            end
            return loss ≤ 0.01f0
        end

        # Warmup
        callback_parameter_estim(θ_init, loss_function(θ_init))
        Zygote.gradient(loss_function, θ_init)

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, θ_init)

        optsol = Optimization.solve(
            optprob, Adam(0.05); callback = callback_parameter_estim,
            maxiters = 1000
        )

        optprob = Optimization.OptimizationProblem(optf, optsol.u)

        optsol = Optimization.solve(
            optprob, Adam(0.001); callback = callback_parameter_estim,
            maxiters = 1000
        )

        θ_estimated = optsol.u

        # Convergence Test
        @test loss_function(θ_estimated) ≤ 0.01f0

        # Run the simulation with the estimated parameters
        @testset "Finite Horizon ODE" begin
            finite_horizon_ode_test(θ_estimated)
        end

        # These controllers have no reason to be stable to infinity
        # Skip those tests here
    end
end

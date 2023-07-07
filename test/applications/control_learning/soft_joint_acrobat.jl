using ComplementaritySolve,
    ComponentArrays,
    DiffEqBase,
    ForwardDiff,
    LinearAlgebra,
    Optimization,
    OptimizationOptimisers,
    OrdinaryDiffEq,
    SciMLSensitivity,
    SimpleNonlinearSolve,
    SparseArrays,
    StableRNGs,
    SteadyStateDiffEq,
    Test,
    Zygote

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

const M = [𝒶+𝒷+2𝒸 𝒷+𝒸; 𝒷+𝒸 𝒷]
const Jᵀ = Float32[-1 1; 0 0]
const A = Float32[0 0 1 0;
    0 0 0 1;
    g/l₁ -(g * m₂)/(l₁ * m₁) 0 0;
    -g/l₁ (g * (l₁ * m₁ + l₁ * m₂ + l₂ * m₂))/(l₁ * l₂ * m₁) 0 0]
const B = reshape(Float32[0
        0
        -(l₁ + l₂) / (l₂ * m₁ * l₁^2)
        (m₁ * l₁^2 + m₂ * (l₁ + l₂)^2) / (m₁ * m₂ * l₁^2 * l₂^2)],
    (4, 1))
const D = vcat(zeros(Float32, 2, 2), inv(M) * Jᵀ)
const a = 0.0f0

const E = sparse([1, 2], [1, 1], Float32[-1, 1], 2, 4)
const F = diagm(Float32[1 / k, 1 / k])
const c = d

rng = StableRNG(0)

x0 = vcat(randn(rng, Float32, 2) .* 0.005f0,
    randn(rng, Float32, 1) * 0.02f0,
    randn(rng, Float32, 1) * 0.01f0)
tspan = (0.0f0, 1.0f0)

controller(x, λ, p, _) = p.K * x .+ p.L * λ

# Taken from https://arxiv.org/pdf/2008.02104.pdf Section 6.C
stable_K = Float32[73.07 38.11 30.41 18.95]
stable_L = Float32[-4.13 4.13]

stable_θ = ComponentArray(; K=stable_K, L=stable_L)

@testset "Stable Controller" begin
    @testset "Finite Horizon ODE" begin
        prob = LCS(x0, controller, tspan, stable_θ, A, B, D, a, E, F, c)
        solver = NaiveLCSAlgorithm(Tsit5(), NonlinearReformulation())
        sol = solve(prob, solver; abstol=1.0f-3, reltol=1.0f-3)

        @test sol isa SciMLBase.ODESolution
        @test SciMLBase.successful_retcode(sol)
        @test all(Base.Fix1(all, isfinite), sol.u)

        @test begin
            ∂stable_θ_ode = only(Zygote.gradient(stable_θ) do θ
                prob = LCS(x0, controller, tspan, θ, A, B, D, a, E, F, c)
                sol = solve(prob,
                    solver;
                    ode_kwargs=(; sensealg=BacksolveAdjoint(; autojacvec=ZygoteVJP())),
                    lcp_kwargs=(; sensealg=LinearComplementarityAdjoint()))
                return sum(abs2, last(sol.u))
            end)

            all(isfinite, ∂stable_θ_ode)
        end
    end

    @testset "Solve to Infinity (Steady-State)" begin
        prob = LCS(x0, controller, (first(tspan), Inf32), stable_θ, A, B, D, a, E, F, c)
        solver = NaiveLCSAlgorithm(DynamicSS(Tsit5();
                termination_condition=NLSolveTerminationCondition(NLSolveTerminationMode.AbsNorm;
                    abstol=1.0f-2,
                    reltol=1.0f-2)),
            NonlinearReformulation())
        sol = solve(prob, solver; abstol=1.0f-3, reltol=1.0f-3)

        @test sol isa SciMLBase.NonlinearSolution
        @test SciMLBase.successful_retcode(sol)
        @test all(isfinite, sol.u)

        @test begin
            ∂stable_θ_ode = only(Zygote.gradient(stable_θ) do θ
                prob = LCS(x0, controller, (first(tspan), Inf32), θ, A, B, D, a, E, F, c)
                sol = solve(prob,
                    solver;
                    ode_kwargs=(; sensealg=SteadyStateAdjoint(; autojacvec=ZygoteVJP())),
                    lcp_kwargs=(; sensealg=LinearComplementarityAdjoint()))
                return sum(abs2, last(sol.u))
            end)

            all(isfinite, ∂stable_θ_ode)
        end
    end
end

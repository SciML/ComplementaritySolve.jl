using Zygote,
    LinearAlgebra,
    SimpleNonlinearSolve,
    OrdinaryDiffEq,
    Optimization,
    OptimizationOptimisers,
    SciMLSensitivity,
    Test,
    ComponentArrays,
    StableRNGs

using ComplementaritySolve

const g = 9.81;
const mp = 0.1;
const mc = 1;
const l = 0.5;
const d1 = 0.1;
const k1 = 10.0;
const k2 = 10.0;

rng = StableRNG(0)
#steady state
x_steady = [0.0, 0.0, 0.0, 0.0]
#initial pos
r_x = rand(rng, 3) .* 2 .- 1
x0 = [10 * r_x[1], 0.0, r_x[2], r_x[3]]

#dynamics of the cartpole system

A = [0.0 0.0 1.0 0.0
    0.0 0.0 0.0 1.0
    0.0 (g * mp)/mc 0.0 0.0
    0.0 g * (mc + mp)/(l * mc) 0.0 0.0]
B = reshape([0.0; 0.0; 1 / mc; 1 / (l * mc)], (4, 1));

D = [zeros(Float64, 3, 2); 1/(l * mp) -1/(l * mp)]
E = [-1.0 l 0.0 0.0; 1.0 -l 0.0 0.0]
F = [1/k1 0.0; 0.0 1/k2]
c = d1;
a = 0.0;#[0.0;0.0;0.0; 0.0 ]

#extract dimension information
n = size(A, 2) #dimension of state space
k = size(B, 2)#dimension of input
m = size(D, 2) #number of contacts

tspan = (0.0, 1.0)

controller(x, λ, p, _) = p.K * x .+ p.L * λ

# Taken from https://arxiv.org/pdf/2008.02104.pdf Section 6.C
stable_K = Float64[3.69 -46.7 3.39 -5.71]
stable_L = Float64[-13.98 13.98]

stable_θ = ComponentArray(; K=stable_K, L=stable_L)

rng = StableRNG(0)
#prob = LCS(x0, controller, tspan, stable_θ, A, B, D, a, E, F, c)
#solver = NaiveLCSAlgorithm(Tsit5(), NonlinearReformulation())
#sol = solve(prob, solver)

@testset "Stable Controller" begin
    @testset "Finite Horizon ODE" begin
        prob = LCS(x0, controller, tspan, stable_θ, A, B, D, a, E, F, c)
        solver = NaiveLCSAlgorithm(Tsit5(), NonlinearReformulation())
        sol = solve(prob, solver)

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
end

@testset "Learn a stabilizing controller" begin
    θ_init = ComponentArray(;
        K=10 * (randn(rng, Float64, (k, n)) .- 0.5),
        L=10 * (rand(rng, Float64, (k, m)) .- 0.5))
    solver = NaiveLCSAlgorithm(Tsit5(), NonlinearReformulation())

    function loss_f(θ)
        prob = LCS(x0, controller, tspan, θ, A, B, D, a, E, F, c)
        lcs_sol = solve(prob, solver; abstol=1e-3, reltol=1e-3)
        return sum(abs2, lcs_sol.u[end]) / n
    end

    iter = 0

    function callback(θ, loss)
        iter += 1
        if iter % 100 == 1 || loss ≤ 0.01
            @info "Parameter Estimation with datapoints" iter=iter loss=loss
        end
        return loss ≤ 0.01
    end

    adtype = Optimization.AutoZygote()

    optf = Optimization.OptimizationFunction((x, p) -> loss_f(x), adtype)

    optprob = Optimization.OptimizationProblem(optf, θ_init)

    result_neurallcs = Optimization.solve(optprob,
        ADAM(0.1);
        callback=callback,
        maxiters=5000)

    optprob2 = Optimization.OptimizationProblem(optf, result_neurallcs.u)

    result_neurallcs2 = Optimization.solve(optprob2,
        ADAM(0.001);
        callback=callback,
        maxiters=10000)

    θ_estimated = result_neurallcs2.u

    # Convergence Test
    @test loss_f(θ_estimated) ≤ 0.5
end

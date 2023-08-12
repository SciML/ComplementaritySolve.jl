using Zygote, LinearAlgebra, SimpleNonlinearSolve, OrdinaryDiffEq, Optimization,
    OptimizationOptimisers, SciMLSensitivity, Test, ComponentArrays, SparseArrays,
    StableRNGs
using ComplementaritySolve

# parameters 
const m1 = 1.0;
const m2 = 1.0;
const m3 = 1.0;
const g = 9.81;
const mp = 1.5;
const l = 0.5;
const k1 = 0.01;

rng = StableRNG(0)
# dynamics of the partial feedback cartpole system
A = sparse([1, 2, 3, 4, 5, 8],
    [5, 6, 7, 8, 4, 4],
    [1.0, 1.0, 1.0, 1.0, g * mp / m1, g * (m1 + mp) / (m1 * l)])
B = sparse([5, 7, 8], [1, 2, 1], [1 / m1, 1 / m3, 1 / (m1 * l)])
D = sparse([5, 6, 6, 7, 8],
    [1, 1, 2, 2, 1],
    [-1 / m1, 1 / m2, -1 / m2, 1 / m3, -1 / (m1 * l)])
a = 0.0
E = [-1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 -1.0 1.0 0.0 0.0 0.0 0.0 0.0]
E = sparse(E)
F = spdiagm([1 / k1, 1 / k1])
c = 0.0

# steady state
x_steady = [0.0, 0.0, 0.0, 0.0]
# initial pos
r_x = rand(rng, 3) .* 2 .- 1
x0 = [10 * r_x[1], 0.0, r_x[2], r_x[3], 0.0, 0.0, 0.0, 0.0]

# extract dimension information
n = size(A, 2) # dimension of state space
k = size(B, 2) # dimension of input
m = size(D, 2) # number of contacts

tspan = (0.0, 1.0)

rng = StableRNG(0)

tspan = (0.0, 1.0)

# Taken from https://arxiv.org/pdf/2008.02104.pdf Section 6.C
stable_K = [-2.8, 6.6, -263.1, 6.4, -2.1, -30.2, 11.5, -12.1, 12.1, 2.6, -4.7, 6.6]
stable_L = [-3.7 -0.6; -0.6 7.2]

stable_θ = ComponentArray(; K=stable_K, L=stable_L)

function controller(x, λ, p, _)
    K = reshape(vcat(p.K[1], 0.0, p.K[2:4], 0.0, p.K[5:7], 0.0, p.K[8:10], 0.0, p.K[11:12]),
        (k, n))
    return K * x .+ p.L * λ
end

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
                sol = solve(prob, solver;
                    ode_kwargs=(; sensealg=BacksolveAdjoint(; autojacvec=ZygoteVJP())),
                    lcp_kwargs=(; sensealg=LinearComplementarityAdjoint()))
                return sum(abs2, last(sol.u))
            end)

            all(isfinite, ∂stable_θ_ode)
        end
    end
end

@testset "Learn a stabilizing controllerish" begin
    θ_init = ComponentArray(; K=randn(rng, Float64, 12), L=rand(rng, Float64, (k, m)))
    solver = NaiveLCSAlgorithm(Tsit5(), NonlinearReformulation())

    function loss_f(θ)
        prob = LCS(x0, controller, tspan, θ, A, B, D, a, E, F, c)
        lcs_sol = solve(prob, solver; abstol=1e-3, reltol=1e-3)
        return sum(abs2, lcs_sol.u[end]) / n
    end

    prob = LCS(x0, controller, tspan, θ_init, A, B, D, a, E, F, c)
    lcs_sol = solve(prob, solver; abstol=1e-3, reltol=1e-3)

    iter = 0

    function callback(θ, loss)
        iter += 1
        if iter % 100 == 1 || loss ≤ 0.1
            @info "Learning a Stabilizing Controller-ish" iter=iter loss=loss
        end
        return loss ≤ 0.1
    end

    adtype = Optimization.AutoZygote()

    optf = Optimization.OptimizationFunction((x, p) -> loss_f(x), adtype)

    optprob = Optimization.OptimizationProblem(optf, θ_init)

    result_neurallcs = Optimization.solve(optprob, Adam(0.1); callback, maxiters=5000)

    optprob2 = Optimization.OptimizationProblem(optf, result_neurallcs.u)

    result_neurallcs2 = Optimization.solve(optprob2, Adam(0.01); callback, maxiters=25000)

    θ_estimated = result_neurallcs2.u

    # Convergence Test
    @test loss_f(θ_estimated) ≤ 0.1
end

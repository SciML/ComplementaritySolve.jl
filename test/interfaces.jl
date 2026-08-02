using ChainRulesCore, ComplementaritySolve, SciMLBase, Test

struct ExternalLinearProblem{M, Q, U} <:
    ComplementaritySolve.AbstractLinearComplementarityProblem{true, false}
    M::M
    q::Q
    u0::U
end

struct ExternalNonlinearProblem{F, U, P} <:
    ComplementaritySolve.AbstractNonlinearComplementarityProblem{false}
    f::F
    u0::U
    p::P
end

struct ExternalSystem <: ComplementaritySolve.AbstractComplementaritySystem{false}
    x0::Vector{Float64}
    λ0::Vector{Float64}
    tspan::Tuple{Float64, Float64}
    p::Nothing
    controller::Function
end

struct ExternalProblemAlgorithm <: ComplementaritySolve.AbstractComplementarityAlgorithm end
struct ExternalSystemAlgorithm <: ComplementaritySolve.AbstractComplementaritySystemAlgorithm end
struct ExternalAdjoint <: ComplementaritySolve.AbstractComplementaritySensitivityAlgorithm end

struct ExternalSolution <: ComplementaritySolve.AbstractLinearComplementaritySolution
    u::Vector{Float64}
    residual::Float64
    prob
    alg
    retcode::SciMLBase.ReturnCode.T
end

function ComplementaritySolve.__solve(
        prob::ExternalLinearProblem, alg::ExternalProblemAlgorithm, u0, M, q; kwargs...
    )
    return ExternalSolution(u0, sum(abs2, M * u0 + q), prob, alg, SciMLBase.ReturnCode.Success)
end

function ComplementaritySolve.__solve(
        prob::ExternalNonlinearProblem, alg::ExternalProblemAlgorithm, u0, p; kwargs...
    )
    return ExternalSolution(u0, sum(abs2, prob.f(u0, p)), prob, alg, SciMLBase.ReturnCode.Success)
end

function ComplementaritySolve.__solve(
        prob::ExternalSystem, alg::ExternalSystemAlgorithm; kwargs...
    )
    return (; prob, alg, kwargs)
end

function ComplementaritySolve.__solve_adjoint(
        ::ExternalLinearProblem, ::ExternalAdjoint, ::ExternalSolution, ::Nothing, args...; kwargs...
    )
    return ntuple(_ -> ChainRulesCore.NoTangent(), length(args))
end

@testset "Generic developer interfaces" begin
    linear_prob = ExternalLinearProblem([2.0;;], [-1.0], [0.0])
    nonlinear_prob = ExternalNonlinearProblem((u, p) -> u .- p, [0.0], [1.0])
    problem_alg = ExternalProblemAlgorithm()

    @test SciMLBase.isinplace(linear_prob)
    @test !ComplementaritySolve.isbatched(linear_prob)
    @test !SciMLBase.isinplace(nonlinear_prob)

    linear_sol = ComplementaritySolve.solve(
        linear_prob, problem_alg; sensealg = ExternalAdjoint(), u0 = [0.5]
    )
    nonlinear_sol = ComplementaritySolve.solve(
        nonlinear_prob, problem_alg; sensealg = ExternalAdjoint(), p = [0.5]
    )
    @test linear_sol.u == [0.5]
    @test nonlinear_sol.residual == 0.25
    @test occursin("retcode", sprint(show, MIME"text/plain"(), linear_sol))

    system_prob = ExternalSystem([0.0], [0.0], (0.0, 1.0), nothing, (x, λ, p, t) -> x)
    system_sol = ComplementaritySolve.solve(system_prob, ExternalSystemAlgorithm(); saveat = 0.1)
    @test system_sol.prob === system_prob
    @test system_sol.kwargs[:saveat] == 0.1

    tangents = ComplementaritySolve.__solve_adjoint(
        linear_prob, ExternalAdjoint(), linear_sol, nothing, linear_sol.u, linear_prob.M, linear_prob.q
    )
    @test tangents == (
        ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
    )
end

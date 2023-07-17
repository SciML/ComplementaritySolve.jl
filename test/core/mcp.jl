using ComplementaritySolve
using ComponentArrays,
    FiniteDifferences,
    ForwardDiff,
    NonlinearSolve,
    SimpleNonlinearSolve,
    StableRNGs,
    Test,
    Zygote

rng = StableRNG(0)

@testset "MCPs" begin
    @testset "Basic MCPs" begin
        # Taken from ParametricMCPs.jl
        f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
        function f!(y, z, θ)
            y[1:2] .= 2z[1:2] - z[3:4] - 2θ
            y[3:4] .= z[1:2]
            return
        end

        feasible_parameters = [
            Float32[0.0, 0.0],
            # Float32[1.0, 0.0],  # Doesn't work yet
            # Float32[0.0, 1.0],  # Doesn't work yet
            [rand(rng, Float32, 2) for _ in 1:10]...,
        ]

        u0 = randn(rng, Float32, 4)
        lower_bound = Float32[-Inf, -Inf, 0, 0]
        upper_bound = Float32[Inf, Inf, Inf, Inf]

        @testset "θ: $(θ)" for θ in feasible_parameters
            @testset "Problem Function: $(func)" for func in (f, f!)
                prob = MCP(func, u0, lower_bound, upper_bound, θ)

                @testset "Solver: $(typeof(solver))" for solver in (PATHSolverAlgorithm(),
                    NonlinearReformulation(:smooth, NewtonRaphson()),
                    NonlinearReformulation(:minmax, NewtonRaphson()))
                    sol = solve(prob, solver)

                    @test sol.u[1:2]≈θ atol=1e-4 rtol=1e-4
                end
            end
        end

        @testset "Adjoint Tests" begin
            function loss_function(θ, solver)
                prob = MCP(f, u0, lower_bound, upper_bound, θ)
                sol = solve(prob, solver; sensealg=MixedComplementarityAdjoint())
                return sum(sol.u)
            end

            loss_function_path = Base.Fix2(loss_function, PATHSolverAlgorithm())
            loss_function_nr = Base.Fix2(loss_function,
                NonlinearReformulation(:smooth, NewtonRaphson()))

            θ = [2.0, -3.0]
            ∂θ_zygote = only(Zygote.gradient(loss_function_path, θ))
            (∂θ_finitediff,) = FiniteDifferences.grad(central_fdm(3, 1),
                loss_function_path,
                θ)
            # FD cant differentiate through the PATH solver (C Code)
            ∂θ_forwarddiff = ForwardDiff.gradient(loss_function_nr, θ)

            @test ∂θ_zygote≈∂θ_forwarddiff atol=1e-3 rtol=1e-3
            @test ∂θ_zygote≈∂θ_finitediff atol=1e-3 rtol=1e-3
        end
    end
end

using ComplementaritySolve
using ComponentArrays,
    ForwardDiff, NonlinearSolve, SimpleNonlinearSolve, StableRNGs, Test, Zygote, PATHSolver

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

        @testset "Nonlinear Reformulation: $(method)" for method in (:smooth, :minmax)
            solver = NonlinearReformulation(method, NewtonRaphson())
            @testset "θ: $(θ)" for θ in feasible_parameters
                @testset "out-of-place" begin
                    # FIXME: Default to unbatched
                    prob = MCP{false, false}(f, u0, lower_bound, upper_bound, θ)
                    sol = solve(prob, solver)

                    @test sol.u[1:2]≈θ atol=1e-4 rtol=1e-4
                end

                @testset "in-place" begin
                    # FIXME: Default to unbatched
                    prob = MCP{true, false}(f!, u0, lower_bound, upper_bound, θ)
                    sol = solve(prob, solver)

                    @test sol.u[1:2]≈θ atol=1e-4 rtol=1e-4
                end
            end
        end

        @testset "Pathsolver" begin
            # https://github.com/chkwon/PATHSolver.jl/blob/master/src/C_API.jl#L459
            @testset "θ: $(θ)" for θ in feasible_parameters
                @testset "out of place" begin
                    solver = PathSolverAlgorithm()
                    prob = MCP{false, false}(f, u0, lower_bound, upper_bound, θ)
                    sol = solve(prob, solver)
                    @test sol.u[1:2]≈θ atol=1e-4 rtol=1e-4
                end
            end
        end
    end
end

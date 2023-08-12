using BenchmarkTools, ComplementaritySolve, ComponentArrays, FiniteDifferences
using ForwardDiff, NonlinearSolve, SimpleNonlinearSolve, StableRNGs, Test, Zygote
import ParametricMCPs

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
                    NonlinearReformulation(:smooth), NonlinearReformulation(:minmax))
                    sol = solve(prob, solver; verbose=false)

                    @test sol.u[1:2]≈θ atol=1e-4 rtol=1e-4
                end
            end
        end

        @testset "Adjoint Tests" begin
            function loss_function(θ, solver)
                prob = MCP(f, u0, lower_bound, upper_bound, θ)
                sol = solve(prob, solver; sensealg=MixedComplementarityAdjoint(),
                    verbose=false)
                return sum(sol.u)
            end

            loss_function_path = Base.Fix2(loss_function, PATHSolverAlgorithm())
            loss_function_nr = Base.Fix2(loss_function, NonlinearReformulation(:smooth))

            θ = [2.0, -3.0]
            ∂θ_zygote = only(Zygote.gradient(loss_function_path, θ))
            (∂θ_finitediff,) = FiniteDifferences.grad(central_fdm(3, 1), loss_function_path,
                θ)
            # FD cant differentiate through the PATH solver (C Code)
            ∂θ_forwarddiff = ForwardDiff.gradient(loss_function_nr, θ)

            @test ∂θ_zygote≈∂θ_forwarddiff atol=1e-3 rtol=1e-3
            @test ∂θ_zygote≈∂θ_finitediff atol=1e-3 rtol=1e-3
        end

        @testset "Benchmarking against ParametricMCPs.jl" begin
            u0 = randn(rng, Float64, 4)
            lb = Float64[-Inf, -Inf, 0, 0]
            ub = Float64[Inf, Inf, Inf, Inf]
            θ = [2.0, -3.0]

            prob_oop = MCP(f, u0, lb, ub, θ)
            function loss_function_oop(θ, solver)
                sol = solve(prob_oop, solver; p=θ, verbose=false)
                return sum(abs2, sol.u)
            end

            prob_iip = MCP(f!, u0, lb, ub, θ)
            function loss_function_iip(θ, solver)
                sol = solve(prob_iip, solver; p=θ, verbose=false)
                return sum(abs2, sol.u)
            end

            prob_ext = ParametricMCPs.ParametricMCP(f, lb, ub, length(θ))
            function loss_function_parametric_mcp(θ)
                sol = ParametricMCPs.solve(prob_ext, θ)
                return sum(abs2, sol.z)
            end

            function loss_function_parametric_mcp_total(θ)
                prob = Zygote.@ignore ParametricMCPs.ParametricMCP(f, lb, ub, length(θ))
                sol = ParametricMCPs.solve(prob, θ)
                return sum(abs2, sol.z)
            end

            loss_function_path_oop = Base.Fix2(loss_function_oop, PATHSolverAlgorithm())
            loss_function_nr_oop = Base.Fix2(loss_function_oop,
                NonlinearReformulation(:smooth))
            loss_function_path_iip = Base.Fix2(loss_function_iip, PATHSolverAlgorithm())
            loss_function_nr_iip = Base.Fix2(loss_function_iip,
                NonlinearReformulation(:smooth))

            for loss_function in (loss_function_path_oop,
                loss_function_nr_oop,
                loss_function_path_iip,
                loss_function_nr_iip)
                t₁ = @belapsed $loss_function($θ)
                t₂ = @belapsed only(Zygote.gradient($loss_function, $θ))
                @info "ComplementaritySolve.jl: $(loss_function)" fwd_time=t₁ with_adjoint_time=t₂
            end

            for loss_function in (loss_function_parametric_mcp,
                loss_function_parametric_mcp_total)
                t₁ = @belapsed $loss_function($θ)
                t₂ = @belapsed only(Zygote.gradient($loss_function, $θ))
                @info "ParametricMCPs.jl: $(loss_function)" fwd_time=t₁ with_adjoint_time=t₂
            end
        end
    end
end

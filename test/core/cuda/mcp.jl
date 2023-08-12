using BenchmarkTools, ComplementaritySolve, ComponentArrays, FiniteDifferences
using ForwardDiff, NonlinearSolve, SimpleNonlinearSolve, StableRNGs, Test, Zygote

using CUDA
CUDA.allowscalar(false)

rng = StableRNG(0)

@testset "MCPs" begin
    @testset "Basic MCPs" begin
        # Taken from ParametricMCPs.jl
        f(z, θ) = vcat(2z[1:2] - z[3:4] - 2θ, z[1:2])
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
        ] .|> cu

        u0 = randn(rng, Float32, 4) |> cu
        lower_bound = Float32[-Inf, -Inf, 0, 0] |> cu
        upper_bound = Float32[Inf, Inf, Inf, Inf] |> cu

        @testset "θ: $(θ)" for θ in feasible_parameters
            @testset "Problem Function: $(func)" for func in (f, f!)
                prob = MCP(func, u0, lower_bound, upper_bound, θ)
                @testset "Solver: $(typeof(solver))" for solver in (NonlinearReformulation(:smooth),
                    NonlinearReformulation(:minmax))
                    sol = solve(prob, solver; verbose=false)

                    @test sol.u[1:2]≈θ atol=1e-4 rtol=1e-4
                end
            end
        end

        @testset "Adjoint Tests" begin
            function loss_function(θ)
                prob = MCP(f, u0, lower_bound, upper_bound, θ)
                sol = solve(prob, NonlinearReformulation(:smooth);
                    sensealg=MixedComplementarityAdjoint(), verbose=false)
                return sum(sol.u)
            end

            function loss_function_cpu(θ)
                prob = MCP(f, u0 |> Array, lower_bound |> Array, upper_bound |> Array, θ)
                sol = solve(prob, NonlinearReformulation(:smooth); verbose=false)
                return sum(sol.u)
            end

            θ = [2.0, -3.0] |> cu
            ∂θ_zygote = only(Zygote.gradient(loss_function, θ))
            ∂θ_forwarddiff = ForwardDiff.gradient(loss_function_cpu, Array(θ))

            @test Array(∂θ_zygote)≈∂θ_forwarddiff atol=1e-6 rtol=1e-6
        end
    end
end

using ComplementaritySolve, ComponentArrays, FiniteDifferences, ForwardDiff
using SimpleNonlinearSolve, StableRNGs, Test, Zygote

using CUDA
CUDA.allowscalar(false)

rng = StableRNG(0)

include("../../test_utils.jl")

@testset "LCPs" begin
    @testset "Basic LCPs" begin
        # https://optimization.cbe.cornell.edu/index.php?title=Linear_complementarity_problem
        A = [2.0 1; 1 2.0] |> cu
        q = [-5.0, -6] |> cu

        prob = LinearComplementarityProblem(A, q)

        @testset "solver: $(nameof(typeof(solver)))" for solver in [InteriorPointMethod(),
            NonlinearReformulation(), BokhovenIterativeAlgorithm()]
            sol = solve(prob, solver)

            u = Array(sol.u)
            @test u≈[4.0 / 3, 7.0 / 3] rtol=1e-3
            w = Array(A * sol.u .+ q)
            @test w≈[0.0, 0.0] atol=1e-3
        end

        @testset "Batched Version" begin
            prob = LinearComplementarityProblem(A, q, rand(rng, 2, 4) |> cu)

            @testset "solver: $(nameof(typeof(solver)))" for solver in [
                BokhovenIterativeAlgorithm(), InteriorPointMethod(),
                NonlinearReformulation()]
                sol = solve(prob, solver)

                @test all(z -> ≈(Array(z), [4.0 / 3, 7.0 / 3]; rtol=1e-3), eachcol(sol.u))
                @test all(z -> ≈(Array(A * z .+ q), [0.0, 0.0]; atol=1e-3), eachcol(sol.u))
            end
        end

        @testset "Adjoint Basic Test" begin
            @testset "size(u0): $sz" for sz in ((2,), (2, 3))
                u0 = rand(StableRNG(0), sz...) |> cu
                solver = NonlinearReformulation()

                for loss_function in (sum, Base.Fix1(sum, abs2))
                    ∂A, ∂q = Zygote.gradient(A, q) do A, q
                        prob = LinearComplementarityProblem{false}(A, q, u0)
                        sol = solve(prob, solver)
                        return loss_function(sol.u)
                    end

                    θ = ComponentArray((; A=Array(A), q=Array(q)))
                    ∂θ_fd = ForwardDiff.gradient(θ) do θ
                        prob = LinearComplementarityProblem{false}(θ.A, θ.q, Array(u0))
                        sol = solve(prob, solver)
                        return loss_function(sol.u)
                    end

                    @test ∂A !== nothing && !iszero(∂A) && size(∂A) == size(A)
                    @test ∂q !== nothing && !iszero(∂q) && size(∂q) == size(q)
                    @test Array(∂A)≈∂θ_fd.A atol=1e-3 rtol=1e-3
                    @test Array(∂q)≈∂θ_fd.q atol=1e-3 rtol=1e-3

                    @test_nowarn Zygote.gradient(A, q) do M, q
                        prob = LinearComplementarityProblem{true}(M, q, u0)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end
                end
            end

            __sizes_list = (((2, 2, 5), (2, 5)), ((2, 2), (2, 5)), ((2, 2, 1), (2, 5)),
                ((2, 2, 5), (2, 1)), ((2, 2, 5), (2, 5)))
            @testset "Batched Adjoint Problem: size(M) = $(szM), size(q) = $(szq)" for (szM, szq) in __sizes_list
                M_ = rand(rng, Float32, szM...) |> cu
                q_ = randn(rng, Float32, szq...) |> cu

                for loss_function in (sum, Base.Fix1(sum, abs2))
                    ∂M, ∂q = Zygote.gradient(M_, q_) do M, q
                        prob = LinearComplementarityProblem{false}(M, q)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end

                    θ = ComponentArray((; M=Array(M_), q=Array(q_)))
                    ∂θ_fd = ForwardDiff.gradient(θ) do θ
                        prob = LinearComplementarityProblem{false}(θ.M, θ.q)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end

                    @test ∂M !== nothing && size(∂M) == size(M_)
                    @test ∂q !== nothing && size(∂q) == size(q_)
                    @test Array(∂M)≈∂θ_fd.M atol=1e-2 rtol=1e-2
                    @test Array(∂q)≈∂θ_fd.q atol=1e-2 rtol=1e-2

                    # We can't check for correctness with FwdDiff & FiniteDifferences
                    # for inplace problems since the in-place batched solvers are not as
                    # accurate as the non-inplace ones.
                    @test_nowarn Zygote.gradient(M_, q_) do M, q
                        prob = LinearComplementarityProblem{true}(M, q)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end
                end
            end
        end
    end
end

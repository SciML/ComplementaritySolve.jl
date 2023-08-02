using ComplementaritySolve, ComponentArrays, FiniteDifferences, ForwardDiff
using SimpleNonlinearSolve, StableRNGs, Test, Zygote

rng = StableRNG(0)

include("utils.jl")

@testset "LCPs" begin
    @testset "Basic LCPs" begin
        # https://optimization.cbe.cornell.edu/index.php?title=Linear_complementarity_problem
        A = [2.0 1; 1 2.0]
        q = [-5.0, -6]

        prob = LinearComplementarityProblem(A, q, zeros(2))

        @testset "solver: $(nameof(typeof(solver)))" for solver in [
            BokhovenIterativeAlgorithm(),
            RPSOR(; ω=1.0, ρ=0.1),
            PGS(),
            InteriorPointMethod(),
        ]
            sol = solve(prob, solver)

            @test sol.u≈[4.0 / 3, 7.0 / 3] rtol=1e-3
            w = A * sol.u .+ q
            @test w≈[0.0, 0.0] atol=1e-6
        end

        @testset "Batched Version" begin
            prob = LinearComplementarityProblem(A, q, rand(rng, 2, 4))

            @testset "solver: $(nameof(typeof(solver)))" for solver in [
                BokhovenIterativeAlgorithm(),
                RPGS(),
                PGS(),
                PSOR(),
                NonlinearReformulation(:smooth, SimpleDFSane(; batched=true)),
                InteriorPointMethod(),
            ]
                sol = solve(prob, solver)

                @test all(z -> ≈(z, [4.0 / 3, 7.0 / 3]; rtol=1e-3), eachcol(sol.u))
                @test all(z -> ≈(A * z .+ q, [0.0, 0.0]; atol=1e-3), eachcol(sol.u))
            end
        end

        @testset "Adjoint Basic Test" begin
            @testset "size(u0): $sz" for sz in ((2,), (2, 3))
                u0 = rand(StableRNG(0), sz...)
                solver = NonlinearReformulation(:smooth,
                    SimpleNewtonRaphson(; batched=true))

                for loss_function in (sum, Base.Fix1(sum, abs2))
                    ∂A, ∂q = Zygote.gradient(A, q) do A, q
                        prob = LinearComplementarityProblem{false}(A, q, u0)
                        sol = solve(prob, solver)
                        return loss_function(sol.u)
                    end

                    θ = ComponentArray((; A, q))
                    ∂θ_fd = ForwardDiff.gradient(θ) do θ
                        prob = LinearComplementarityProblem{false}(θ.A, θ.q, u0)
                        sol = solve(prob, solver)
                        return loss_function(sol.u)
                    end

                    (∂θ_finitediff,) = FiniteDifferences.grad(central_fdm(3, 1),
                        θ -> begin
                            prob = LinearComplementarityProblem{false}(θ.A, θ.q, u0)
                            sol = solve(prob, PGS())
                            return loss_function(sol.u)
                        end,
                        θ)

                    @test ∂A !== nothing && !iszero(∂A) && size(∂A) == size(A)
                    @test ∂q !== nothing && !iszero(∂q) && size(∂q) == size(q)
                    @test ∂A≈∂θ_fd.A atol=1e-3 rtol=1e-3
                    @test ∂q≈∂θ_fd.q atol=1e-3 rtol=1e-3
                    @test ∂A≈∂θ_finitediff.A atol=1e-3 rtol=1e-3
                    @test ∂q≈∂θ_finitediff.q atol=1e-3 rtol=1e-3

                    @test_nowarn Zygote.gradient(A, q) do M, q
                        prob = LinearComplementarityProblem{true}(M, q, u0)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end
                end
            end

            __sizes_list = (((2, 2, 5), (2, 5)),
                ((2, 2), (2, 5)),
                ((2, 2, 1), (2, 5)),
                ((2, 2, 5), (2, 1)),
                ((2, 2, 5), (2, 5)))
            @testset "Batched Adjoint Problem: size(M) = $(szM), size(q) = $(szq)" for (szM, szq) in __sizes_list
                M_ = rand(rng, Float32, szM...)
                q_ = randn(rng, Float32, szq...)

                for loss_function in (sum, Base.Fix1(sum, abs2))
                    ∂M, ∂q = Zygote.gradient(M_, q_) do M, q
                        prob = LinearComplementarityProblem{false}(M, q)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end

                    θ = ComponentArray((; M=M_, q=q_))
                    ∂θ_fd = ForwardDiff.gradient(θ) do θ
                        prob = LinearComplementarityProblem{false}(θ.M, θ.q)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end

                    (∂θ_finitediff,) = FiniteDifferences.grad(central_fdm(3, 1),
                        θ -> begin
                            prob = LinearComplementarityProblem{false}(θ.M, θ.q)
                            sol = solve(prob)
                            return loss_function(sol.u)
                        end,
                        θ)

                    @test ∂M !== nothing && size(∂M) == size(M_)
                    @test ∂q !== nothing && size(∂q) == size(q_)
                    @test ∂M≈∂θ_fd.M atol=1e-3 rtol=1e-3
                    @test ∂q≈∂θ_fd.q atol=1e-3 rtol=1e-3
                    @test ∂M≈∂θ_finitediff.M atol=1e-3 rtol=1e-3
                    @test ∂q≈∂θ_finitediff.q atol=1e-3 rtol=1e-3

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

    @testset "Convergence Test" begin
        #taken from https://github.com/siconos/siconos/tree/master/numerics/src/LCP/test/data
        @testset "Test PGS" begin
            file_names = [
                "./data/lcp_CPS_5.dat",
                "./data/lcp_CPS_5.dat",
                "./data/lcp_exp_murty.dat",
                "./data/lcp_exp_murty2.dat",
            ]
            test_data = [parse_data(file_name) for file_name in file_names]
            for (M, q) in test_data
                prob = LinearComplementarityProblem(M, q, rand(rng, length(q)))
                sol = solve(prob, PGS())
                @test all(isfinite, sol.u)
            end
        end

        @testset "NonlinearReformulation" begin
            @testset "Netwon-Raphson" begin
                file_names = ["./data/lcp_ortiz.dat", "./data/lcp_CPS_2.dat"]
                test_data = [parse_data(file_name) for file_name in file_names]
                for (M, q) in test_data
                    prob = LinearComplementarityProblem(M, q, rand(rng, length(q)))
                    sol = solve(prob, NonlinearReformulation())
                    @test all(isfinite, sol.u)
                end
            end

            @testset "Broyden" begin
                file_names = [
                    "./data/lcp_CPS_3.dat",
                    "./data/lcp_CPS_4.dat",
                    "./data/lcp_CPS_4bis.dat",
                    "./data/lcp_Pang_isolated_sol.dat",
                    "./data/lcp_Pang_isolated_sol_perturbed.dat",
                ]
                test_data = [parse_data(file_name) for file_name in file_names]

                for (M, q) in test_data
                    prob = LinearComplementarityProblem(M, q, rand(rng, length(q)))
                    sol = solve(prob,
                        NonlinearReformulation(:smooth, Broyden(; batched=true)))
                    @test all(isfinite, sol.u)
                end
            end
        end

        @testset "BokhovenIterative and IPM test" begin
            file_names = ["./data/lcp_trivial.dat", "./data/lcp_mmc.dat"]
            test_positive = [parse_data(file_name) for file_name in file_names]
            solvers = [BokhovenIterativeAlgorithm(), InteriorPointMethod()]

            for (M, q) in test_positive
                prob = LinearComplementarityProblem(M, q, rand(rng, length(q)))
                for solver in solvers
                    sol = solve(prob, solver)
                    @test all(isfinite, sol.u)
                end
            end
        end
    end
end

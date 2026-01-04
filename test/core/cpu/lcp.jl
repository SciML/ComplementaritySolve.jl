using ComplementaritySolve, ComponentArrays, FiniteDifferences, ForwardDiff
using SimpleNonlinearSolve, StableRNGs, Test, Zygote

rng = StableRNG(0)

include("../../test_utils.jl")

@testset "LCPs" begin
    @testset "Basic LCPs" begin
        # https://optimization.cbe.cornell.edu/index.php?title=Linear_complementarity_problem
        A = [2.0 1; 1 2.0]
        q = [-5.0, -6]

        prob = LinearComplementarityProblem(A, q, zeros(2))

        @testset "solver: $(nameof(typeof(solver)))" for solver in [
                RPSOR(; ω = 1.0, ρ = 0.1),
                BokhovenIterativeAlgorithm(), PGS(), InteriorPointMethod(),
            ]
            sol = solve(prob, solver)

            @test sol.u ≈ [4.0 / 3, 7.0 / 3] rtol = 1.0e-3
            w = A * sol.u .+ q
            @test w ≈ [0.0, 0.0] atol = 1.0e-6
        end

        @testset "Batched Version" begin
            prob = LinearComplementarityProblem(A, q, rand(rng, 2, 4))

            @testset "solver: $(nameof(typeof(solver)))" for solver in [
                    RPGS(), PGS(),
                    PSOR(), BokhovenIterativeAlgorithm(), InteriorPointMethod(),
                    NonlinearReformulation(:smooth, SimpleDFSane(; batched = true)),
                ]
                sol = solve(prob, solver)

                @test all(z -> ≈(z, [4.0 / 3, 7.0 / 3]; rtol = 1.0e-3), eachcol(sol.u))
                @test all(z -> ≈(A * z .+ q, [0.0, 0.0]; atol = 1.0e-3), eachcol(sol.u))
            end
        end

        @testset "Adjoint Basic Test" begin
            @testset "size(u0): $sz" for sz in ((2,), (2, 3))
                u0 = rand(StableRNG(0), sz...)
                solver = NonlinearReformulation(
                    :smooth,
                    SimpleNewtonRaphson(; batched = true)
                )

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

                    (∂θ_finitediff,) = FiniteDifferences.grad(
                        central_fdm(3, 1),
                        θ -> begin
                            prob = LinearComplementarityProblem{false}(θ.A, θ.q, u0)
                            sol = solve(prob, PGS())
                            return loss_function(sol.u)
                        end,
                        θ
                    )

                    @test ∂A !== nothing && !iszero(∂A) && size(∂A) == size(A)
                    @test ∂q !== nothing && !iszero(∂q) && size(∂q) == size(q)
                    @test ∂A ≈ ∂θ_fd.A atol = 1.0e-3 rtol = 1.0e-3
                    @test ∂q ≈ ∂θ_fd.q atol = 1.0e-3 rtol = 1.0e-3
                    @test ∂A ≈ ∂θ_finitediff.A atol = 1.0e-3 rtol = 1.0e-3
                    @test ∂q ≈ ∂θ_finitediff.q atol = 1.0e-3 rtol = 1.0e-3

                    @test_nowarn Zygote.gradient(A, q) do M, q
                        prob = LinearComplementarityProblem{true}(M, q, u0)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end
                end
            end

            __sizes_list = (
                ((2, 2, 5), (2, 5)), ((2, 2), (2, 5)), ((2, 2, 1), (2, 5)),
                ((2, 2, 5), (2, 1)), ((2, 2, 5), (2, 5)),
            )
            @testset "Batched Adjoint Problem: size(M) = $(szM), size(q) = $(szq)" for (
                    szM, szq,
                ) in __sizes_list

                M_ = rand(rng, Float32, szM...)
                q_ = randn(rng, Float32, szq...)

                for loss_function in (sum, Base.Fix1(sum, abs2))
                    ∂M, ∂q = Zygote.gradient(M_, q_) do M, q
                        prob = LinearComplementarityProblem{false}(M, q)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end

                    θ = ComponentArray((; M = M_, q = q_))
                    ∂θ_fd = ForwardDiff.gradient(θ) do θ
                        prob = LinearComplementarityProblem{false}(θ.M, θ.q)
                        sol = solve(prob)
                        return loss_function(sol.u)
                    end

                    (∂θ_finitediff,) = FiniteDifferences.grad(
                        central_fdm(3, 1),
                        θ -> begin
                            prob = LinearComplementarityProblem{false}(θ.M, θ.q)
                            sol = solve(prob)
                            return loss_function(sol.u)
                        end,
                        θ
                    )

                    @test ∂M !== nothing && size(∂M) == size(M_)
                    @test ∂q !== nothing && size(∂q) == size(q_)
                    @test ∂M ≈ ∂θ_fd.M atol = 1.0e-3 rtol = 1.0e-3
                    @test ∂q ≈ ∂θ_fd.q atol = 1.0e-3 rtol = 1.0e-3
                    @test ∂M ≈ ∂θ_finitediff.M atol = 1.0e-3 rtol = 1.0e-3
                    @test ∂q ≈ ∂θ_finitediff.q atol = 1.0e-3 rtol = 1.0e-3

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

    # TODO: Streamline these. Too much code duplication!
    # taken from https://github.com/siconos/siconos/tree/master/numerics/src/LCP/test/data
    @testset "Convergence Test" begin
        @testset "PGS: $(file_name)" for file_name in [
                "data/lcp_CPS_1.dat",
                "data/lcp_CPS_5.dat",
                "data/lcp_exp_murty.dat",
                "data/lcp_exp_murty2.dat",
                "data/lcp_deudeu.dat",
            ]
            (M, q) = parse_lcp_data(joinpath(@__DIR__, "..", file_name))
            prob = LCP(M, q)
            sol = solve(prob, PGS())
            @test all(≥(-1.0e-5), sol.u)
            w = M * sol.u .+ q
            @test (w' * sol.u) ≈ 0.0 atol = 1.0e-6
        end

        @testset "NonlinearReformulation" begin

            # problems that pass with direct_solvers, iterative_solvers, equation-based solvers
            @testset "Netwon-Raphson: $(file_name)" for file_name in [
                    "data/lcp_CPS_2.dat",
                    "data/lcp_CPS_3.dat",
                    "data/lcp_ortiz.dat",
                ]
                (M, q) = parse_lcp_data(joinpath(@__DIR__, "..", file_name))
                prob = LCP(M, q)
                sol = solve(prob, NonlinearReformulation())
                @test all(>=(-1.0e-5), sol.u)
                w = M * sol.u .+ q
                @test (w' * sol.u) ≈ 0.0 atol = 1.0e-6
            end

            @testset "Broyden: $(file_name)" for file_name in [
                    "data/lcp_CPS_3.dat", # iterative solvers test
                    "data/lcp_enum_fails.dat", # direct solver LCP_enum
                ]
                (M, q) = parse_lcp_data(joinpath(@__DIR__, "..", file_name))
                prob = LCP(M, q)
                sol = solve(prob, NonlinearReformulation(:smooth, Broyden(; batched = true)))
                @test all(>=(-1.0e-5), sol.u)
                w = M * sol.u .+ q
                @test (w' * sol.u) ≈ 0.0 atol = 1.0e-6
            end
        end

        @testset "Positive Definite Problems" begin
            # both problems pass with direct_solvers, iterative_solvers, equation-based solvers
            # but M is positive definite so we can use our solvers that target such problems

            @testset "BokhovenIterative: $(file_name)" for file_name in [
                    "data/lcp_trivial.dat",
                    "data/lcp_mmc.dat",
                ]
                (M, q) = parse_lcp_data(joinpath(@__DIR__, "..", file_name))
                prob = LCP(M, q)
                sol = solve(prob, BokhovenIterativeAlgorithm())

                @test all(>=(-1.0e-5), sol.u)
                w = M * sol.u .+ q
                @test (w' * sol.u) ≈ 0.0 atol = 1.0e-6
            end

            @testset "IPM: $(file_name)" for file_name in ["data/lcp_trivial.dat"]
                (M, q) = parse_lcp_data(joinpath(@__DIR__, "..", file_name))
                prob = LCP(M, q)
                sol = solve(prob, InteriorPointMethod())

                @test all(>=(-1.0e-5), sol.u)
                w = M * sol.u .+ q
                @test (w' * sol.u) ≈ 0.0 atol = 1.0e-6
            end
        end

        # the problems were supposed to work with iterative solvers, such as PGS,RPGS,PSOR.
        # Our implementation uses the same default parameters as siconos tests.
        @testset "Iterative Solvers: $(file_name)" for file_name in [
                "data/lcp_CPS_4.dat",
                "data/lcp_CPS_4bis.dat",
            ]
            (M, q) = parse_lcp_data(joinpath(@__DIR__, "..", file_name))
            prob = LCP(M, q)
            sol = solve(prob, RPGS())
            @test_broken all(>=(-1.0e-5), sol.u)
            w = M * sol.u .+ q
            @test_broken (w' * sol.u) ≈ 0.0 atol = 1.0e-6
        end

        #the problems were supposed to pass with Direct Solvers in siconos, such as Lemke
        @testset "Direct Solvers: $(file_name)" for file_name in [
                "data/lcp_Pang_isolated_sol.dat",
                "data/lcp_Pang_isolated_sol_perturbed.dat",
            ]
            (M, q) = parse_lcp_data(joinpath(@__DIR__, "..", file_name))
            prob = MCP(LCP(M, q))
            sol = solve(prob, PATHSolverAlgorithm())
            @test all(>=(-1.0e-5), sol.u)
            w = M * sol.u .+ q
            @test_broken (w' * sol.u) ≈ 0.0 atol = 1.0e-6
        end
    end
end

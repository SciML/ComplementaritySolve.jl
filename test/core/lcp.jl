using ComplementaritySolve, ComponentArrays, FiniteDifferences, ForwardDiff
using SimpleNonlinearSolve, StableRNGs, Test, Zygote

rng = StableRNG(0)

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
            test_data = [
                (; M=[1.0 1.0;
                    1.0 1.0], q=[-1.0, -1.0]),
                (; M=[1.0 -1.0;
                    -1.0 1.0], q=[1.0, -1.0]),#lcp_cps_5.dat
                (;
                    M=[2.000000000000000000000000e+00 1.000000000000000000000000e+00;
                        1.000000000000000000000000e+00 2.000000000000000000000000e+00],
                    q=[-5.000000000000000000000000e+00, -6.000000000000000000000000e+00]),#lcp_deudeu.dat
                (;
                    M=[1.0 0.0 0.0 0.0 0.0 0.0;
                        2.0 1.0 0.0 0.0 0.0 0.0;
                        2.0 2.0 1.0 0.0 0.0 0.0;
                        2.0 2.0 2.0 1.0 0.0 0.0;
                        2.0 2.0 2.0 2.0 1.0 0.0;
                        2.0 2.0 2.0 2.0 2.0 1.0],
                    q=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),#lcp_exp_murty.dat
                (;
                    M=[1.0 0.0 0.0 0.0 0.0 0.0;
                        2.0 1.0 0.0 0.0 0.0 0.0;
                        2.0 2.0 1.0 0.0 0.0 0.0;
                        2.0 2.0 2.0 1.0 0.0 0.0;
                        2.0 2.0 2.0 2.0 1.0 0.0;
                        2.0 2.0 2.0 2.0 2.0 1.0],
                    q=[-126.0, -124.0, -120.0, -112.0, -96.0, -64.0]), #lcp_exp_murty2.dat
            ]

            for (i, (M, q)) in enumerate(test_data)
                prob = LinearComplementarityProblem(M, q, rand(length(q)))
                sol = solve(prob, PGS())
                @test all(isfinite, sol.u)
            end
        end

        @testset "NonlinearReformulation" begin
            @testset "Netwon-Raphson" begin
                test_data = [
                    (; M=[0.0 2.0 -1.0;
                        -1.0 0.0 1.0;
                        2.0 -2.0 0.0], q=[-3.0, 6.0, -1.0]), #lcp_cps_2.dat
                    (;
                        M=[0.0 0.0 10.0 30.0;
                            0.0 0.0 20.0 15.0;
                            10.0 30.0 0.0 0.0;
                            20.0 15.0 0.0 0.0],
                        q=[-1.0, -1.0, -1.0, -1.0]), #lcp_cps_3.dat
                ]

                for (i, (M, q)) in enumerate(test_data)
                    prob = LinearComplementarityProblem(M, q, rand(length(q)))
                    sol = solve(prob, NonlinearReformulation())
                    @test all(isfinite, sol.u)
                end
            end

            @testset "Broyden" begin
                test_data = [
                    (;
                        M=[11.0 0.0 10.0 1.0;
                            0.0 11.0 10.0 1.0;
                            10.0 10.0 21.0 1.0;
                            -1.0 -1.0 -1.0 0.0],
                        q=[50.0, 50.0, 10.0, -6.0]),#lcp_cps_4.dat
                    (;
                        M=[11.0 0.0 10.0 1.0;
                            0.0 11.0 10.0 1.0;
                            10.0 10.0 21.0 1.0;
                            -1.0 -1.0 -1.0 0.0],
                        q=[50.0, 50.0, 23.0, -6.0]),#lcp_cps_4bis.dat
                    (; M=[0.0 1.0 -1.0;
                        -1.0 0.0 0.0;
                        -1.0 0.0 0.0], q=[0.0, -1.0, 1.0]), #lcp_pang_isolated_sol.dat
                ]

                for (i, (M, q)) in enumerate(test_data)
                    prob = LinearComplementarityProblem(M, q, rand(length(q)))
                    sol = solve(prob,
                        NonlinearReformulation(:smooth, Broyden(; batched=true)))
                    @test all(isfinite, sol.u)
                end
            end
        end
    end
end

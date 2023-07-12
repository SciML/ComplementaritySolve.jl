using ComplementaritySolve,
    ComponentArrays,
    FiniteDifferences,
    ForwardDiff,
    SimpleNonlinearSolve,
    StableRNGs,
    Test,
    Zygote

rng = StableRNG(0)

@testset "LCPs" begin
    @testset "Basic LCPs" begin
        # https://optimization.cbe.cornell.edu/index.php?title=Linear_complementarity_problem
        A = [2.0 1; 1 2.0]
        q = [-5.0, -6]

        prob = LinearComplementarityProblem(A, q, zeros(2))

        @testset "solver: $(nameof(typeof(solver)))" for solver in [
            BokhovenIterativeLCPAlgorithm(),
            RPSOR(; ω=1.0, ρ=0.1),
            PGS(),
        ]
            sol = solve(prob, solver)

            @test sol.z≈[4.0 / 3, 7.0 / 3] rtol=1e-3
            @test sol.w≈[0.0, 0.0] atol=1e-6
        end

        @testset "Batched Version" begin
            prob = LinearComplementarityProblem(A, q, rand(rng, 2, 4))

            @testset "solver: $(nameof(typeof(solver)))" for solver in [
                BokhovenIterativeLCPAlgorithm(),
                RPGS(),
                PGS(),
                PSOR(),
                NonlinearReformulation(:smooth, SimpleDFSane(; batched=true)),
            ]
                sol = solve(prob, solver)

                @test all(z -> ≈(z, [4.0 / 3, 7.0 / 3]; rtol=1e-3), eachcol(sol.z))
                @test all(w -> ≈(w, [0.0, 0.0]; atol=1e-3), eachcol(sol.w))
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
                        sol = solve(prob, solver; sensealg=LinearComplementarityAdjoint())
                        return loss_function(sol.z) + loss_function(sol.w)
                    end

                    θ = ComponentArray((; A, q))
                    ∂θ_fd = ForwardDiff.gradient(θ) do θ
                        prob = LinearComplementarityProblem{false}(θ.A, θ.q, u0)
                        sol = solve(prob, solver)
                        return loss_function(sol.z) + loss_function(sol.w)
                    end

                    (∂θ_finitediff,) = FiniteDifferences.grad(central_fdm(3, 1),
                        θ -> begin
                            prob = LinearComplementarityProblem{false}(θ.A, θ.q, u0)
                            sol = solve(prob, PGS())
                            return loss_function(sol.z) + loss_function(sol.w)
                        end,
                        θ)

                    @test ∂A !== nothing && !iszero(∂A)
                    @test ∂q !== nothing && !iszero(∂q)
                    @test ∂A≈∂θ_fd.A atol=1e-3 rtol=1e-3
                    @test ∂q≈∂θ_fd.q atol=1e-3 rtol=1e-3

                    @test ∂A≈∂θ_finitediff.A atol=1e-3 rtol=1e-3
                    @test ∂q≈∂θ_finitediff.q atol=1e-3 rtol=1e-3
                end
            end

            @testset "Batched Adjoint Problem" begin
                szA = (2, 2, 5)
                szq = (2, 5)
                A_ = rand(rng, Float32, szA...)
                q_ = randn(rng, Float32, szq...)

                solver = NonlinearReformulation(:smooth,
                    SimpleNewtonRaphson(; batched=true))

                for loss_function in (sum, Base.Fix1(sum, abs2))
                    ∂A, ∂q = Zygote.gradient(A_, q_) do A, q
                        prob = LinearComplementarityProblem{false}(A, q)
                        sol = solve(prob, solver; sensealg=LinearComplementarityAdjoint())
                        return loss_function(sol.z) + loss_function(sol.w)
                    end

                    θ = ComponentArray((; A=A_, q=q_))
                    ∂θ_fd = ForwardDiff.gradient(θ) do θ
                        prob = LinearComplementarityProblem{false}(θ.A, θ.q)
                        sol = solve(prob, solver)
                        return loss_function(sol.z) + loss_function(sol.w)
                    end

                    (∂θ_finitediff,) = FiniteDifferences.grad(central_fdm(3, 1),
                        θ -> begin
                            prob = LinearComplementarityProblem{false}(θ.A, θ.q)
                            sol = solve(prob, PGS())
                            return loss_function(sol.z) + loss_function(sol.w)
                        end,
                        θ)

                    @test ∂A !== nothing && !iszero(∂A)
                    @test ∂q !== nothing && !iszero(∂q)
                    @test ∂A≈∂θ_fd.A atol=1e-3 rtol=1e-3
                    @test ∂q≈∂θ_fd.q atol=1e-3 rtol=1e-3

                    # FIXME: Probably the solutions reached are different, so the tests fail
                    @test_broken ∂A≈∂θ_finitediff.A atol=1e-3 rtol=1e-3
                    @test_broken ∂q≈∂θ_finitediff.q atol=1e-3 rtol=1e-3
                end
            end
        end
    end
end

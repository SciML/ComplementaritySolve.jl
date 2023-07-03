using ComplementaritySolve, SimpleNonlinearSolve, StableRNGs, Test

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
            prob = LinearComplementarityProblem(A, q, rand(StableRNG(0), 2, 4))

            solver = NonlinearReformulation(:smooth, SimpleDFSane(; batched=true))

            sol = solve(prob, solver)

            true_sol = [4.0 / 3, 7.0 / 3]

            @test all(z -> ≈(z, true_sol; rtol=1e-3), eachcol(sol.z))
            @test all(w -> ≈(w, [0.0, 0.0]; atol=1e-3), eachcol(sol.w))
        end
    end
end

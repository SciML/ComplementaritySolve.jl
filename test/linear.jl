using Test 
using ComplementaritySolve

@testset "Basic LCPs" begin 

    # https://optimization.cbe.cornell.edu/index.php?title=Linear_complementarity_problem
    A = [2.0 1; 1 2.0] 
    q = [-5.0, -6]

    prob = LinearComplementarityProblem(A, q, zeros(2))

    for alg in [BokhovenIterativeLCPAlgorithm(), 
                RPSOR(;ω = 1.0, ρ = 0.1),
                PGS()]
        sol = solve(prob, alg)

        @test sol.z ≈ [4.0/3, 7.0/3] rtol = 1e-3
        @test sol.w ≈ [0.0, 0.0] atol = 1e-6
    end
end  

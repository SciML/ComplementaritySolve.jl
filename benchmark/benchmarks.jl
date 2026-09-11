using ComplementaritySolve, BenchmarkTools
using CommonSolve: solve
using StableRNGs, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Basic LCP: w = A*z + q, w ⟂ z
A_small = [2.0 1.0; 1.0 2.0]
q_small = [-5.0, -6.0]
prob_small = LinearComplementarityProblem(A_small, q_small, zeros(2))

# Larger random LCP (M-matrix for guaranteed solvability)
n = 200
R = rand(rng, n, n)
A_big = R' * R + n * I
q_big = -rand(rng, n)
prob_big = LinearComplementarityProblem(A_big, q_big, zeros(n))

# Batched LCP
prob_batched = LinearComplementarityProblem(A_small, q_small, rand(rng, 2, 8))

# =============================================================================
# LCP solves
# =============================================================================

SUITE["lcp"] = BenchmarkGroup()

SUITE["lcp"]["bokhoven_small"] = @benchmarkable solve(
    $prob_small, BokhovenIterativeAlgorithm()
)
SUITE["lcp"]["pgs_small"] = @benchmarkable solve($prob_small, PGS())
SUITE["lcp"]["rpsor_small"] = @benchmarkable solve(
    $prob_small, RPSOR(; ω = 1.0, ρ = 0.1)
)
SUITE["lcp"]["interior_point_small"] = @benchmarkable solve(
    $prob_small, InteriorPointMethod()
)
SUITE["lcp"]["pgs_200"] = @benchmarkable solve($prob_big, PGS())
SUITE["lcp"]["batched_pgs"] = @benchmarkable solve($prob_batched, PGS())

# =============================================================================
# Problem construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()
SUITE["construct"]["lcp"] = @benchmarkable LinearComplementarityProblem(
    $A_small, $q_small, $(zeros(2))
)

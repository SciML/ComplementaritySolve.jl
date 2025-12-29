using ComplementaritySolve, SparseArrays, Test, Zygote, FiniteDifferences, ForwardDiff

@testset "Sparsity Detection" begin
    @testset "compute_jacobian_sparsity" begin
        # Test with a simple sparse function
        f(z, θ) = [2z[1] - z[2]; z[1] + z[2]]
        u0 = zeros(2)
        θ = nothing

        sparsity = compute_jacobian_sparsity(f, u0, θ)

        # The Jacobian should be:
        # [2 -1]
        # [1  1]
        # Which is fully dense for this case
        @test size(sparsity) == (2, 2)
        @test nnz(sparsity) == 4

        # Test with a sparse function (diagonal Jacobian)
        f_sparse(z, θ) = [z[1]^2; z[2]^2; z[3]^2]
        u0_sparse = ones(3)

        sparsity_sparse = compute_jacobian_sparsity(f_sparse, u0_sparse, nothing)

        # The Jacobian should be diagonal:
        # [2z₁  0    0 ]
        # [ 0  2z₂   0 ]
        # [ 0   0   2z₃]
        @test size(sparsity_sparse) == (3, 3)
        @test nnz(sparsity_sparse) == 3  # Only diagonal elements

        # Test with in-place function
        function f_iip!(y, z, θ)
            y[1] = 2z[1] - z[2]
            y[2] = z[1] + z[2]
            return nothing
        end

        sparsity_iip = compute_jacobian_sparsity(f_iip!, zeros(2), nothing; iip=true)
        @test size(sparsity_iip) == (2, 2)
        @test nnz(sparsity_iip) == 4
    end

    @testset "MCP with automatic sparsity detection" begin
        f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]

        u0 = zeros(4)
        lb = Float64[-Inf, -Inf, 0, 0]
        ub = Float64[Inf, Inf, Inf, Inf]
        θ = [1.0, 2.0]

        # Create MCP with automatic sparsity detection
        prob_sparse = MCP(f, u0, lb, ub, θ, Val(:auto))

        # Check that sparsity pattern was detected
        @test prob_sparse.jac_prototype !== nothing
        @test prob_sparse.jac_prototype isa SparseMatrixCSC

        # The Jacobian of f should be:
        # [ 2  0 -1  0]
        # [ 0  2  0 -1]
        # [ 1  0  0  0]
        # [ 0  1  0  0]
        # This has 6 non-zero entries (2,2,-1,-1,1,1)
        @test nnz(prob_sparse.jac_prototype) == 6

        # Create MCP without sparsity (default)
        prob_dense = MCP(f, u0, lb, ub, θ)
        @test prob_dense.jac_prototype === nothing

        # Create MCP with explicit sparsity pattern
        explicit_pattern = sparse([1, 1, 2, 2, 3, 4], [1, 3, 2, 4, 1, 2],
            ones(6), 4, 4)
        prob_explicit = MCP(f, u0, lb, ub, θ; jac_prototype=explicit_pattern)
        @test prob_explicit.jac_prototype === explicit_pattern
    end

    @testset "Solving MCP with sparsity" begin
        f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]

        u0 = randn(4)
        lb = Float64[-Inf, -Inf, 0, 0]
        ub = Float64[Inf, Inf, Inf, Inf]
        θ = [2.0, -3.0]

        # Create problems with and without sparsity
        prob_sparse = MCP(f, u0, lb, ub, θ, Val(:auto))
        prob_dense = MCP(f, u0, lb, ub, θ)

        # Solve with PATHSolver
        sol_sparse = solve(prob_sparse, PATHSolverAlgorithm(); verbose=false)
        sol_dense = solve(prob_dense, PATHSolverAlgorithm(); verbose=false)

        # Key test: solutions with and without sparsity should be the same
        @test sol_sparse.u ≈ sol_dense.u atol = 1e-6

        # Solve with NonlinearReformulation
        sol_sparse_nr = solve(prob_sparse, NonlinearReformulation(:smooth); verbose=false)
        sol_dense_nr = solve(prob_dense, NonlinearReformulation(:smooth); verbose=false)

        @test sol_sparse_nr.u ≈ sol_dense_nr.u atol = 1e-4
    end

    @testset "Adjoint with sparsity" begin
        f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]

        u0 = randn(4)
        lb = Float64[-Inf, -Inf, 0, 0]
        ub = Float64[Inf, Inf, Inf, Inf]

        function loss_function(θ, use_sparsity)
            if use_sparsity
                prob = MCP(f, u0, lb, ub, θ, Val(:auto))
            else
                prob = MCP(f, u0, lb, ub, θ)
            end
            sol = solve(
                prob, PATHSolverAlgorithm(); sensealg=MixedComplementarityAdjoint(),
                verbose=false)
            return sum(sol.u)
        end

        θ = [2.0, -3.0]

        # Compare gradients with and without sparsity
        ∂θ_sparse = only(Zygote.gradient(θ -> loss_function(θ, true), θ))
        ∂θ_dense = only(Zygote.gradient(θ -> loss_function(θ, false), θ))

        # Gradients should be the same
        @test ∂θ_sparse ≈ ∂θ_dense atol = 1e-4

        # Verify against finite differences
        loss_fn = θ -> loss_function(θ, false)
        (∂θ_fd,) = FiniteDifferences.grad(central_fdm(3, 1), loss_fn, θ)
        @test ∂θ_sparse ≈ ∂θ_fd atol = 1e-3
    end
end

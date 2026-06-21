using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Minimal setup - avoid heavy computations
    # Use small problem sizes for fast precompilation

    @compile_workload begin
        # LCP problem setup and solving
        # This is the most common use case
        A = [2.0 1.0; 1.0 2.0]
        q = [-5.0, -6.0]
        u0 = zeros(2)

        # Create LCP problem (most common entry point)
        prob = LinearComplementarityProblem(A, q, u0)

        # Precompile the most commonly used solvers
        # PGS/RPSOR is fast and commonly used
        sol_pgs = solve(prob, PGS())

        # RPSOR with custom parameters
        sol_rpsor = solve(prob, RPSOR(; ω = 1.0, ρ = 0.1))

        # BokhovenIterativeAlgorithm for positive definite problems
        sol_bokh = solve(prob, BokhovenIterativeAlgorithm())

        # InteriorPointMethod — skip precompilation as it uses FunctionOperator
        # which has API changes across SciMLOperators versions
        # sol_ipm = solve(prob, InteriorPointMethod())

        # NonlinearReformulation with default solver (common path)
        # This exercises the NonlinearSolve integration
        sol_nr = solve(prob, NonlinearReformulation(:smooth))

        # Also precompile minmax variant
        sol_nr_minmax = solve(prob, NonlinearReformulation(:minmax))

        # PATHSolverAlgorithm is intentionally not exercised here. Invoking the
        # PATH native library (libpath50) inside the precompile worker
        # segfaults intermittently, which aborts precompilation of the whole
        # package. Native library calls are not what precompilation caches
        # anyway, so excluding it loses nothing while making precompilation
        # deterministic.
    end
end

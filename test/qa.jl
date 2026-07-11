using SciMLTesting, ComplementaritySolve, Test

run_qa(
    ComplementaritySolve;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = false,    # Too many ambiguities from downstream
        persistent_tasks = false,  # PATHSolver precompile workload triggers persistent task detection
        project_extras = false,
        deps_compat = false,
    ),
    api_docs_kwargs = (;
        ignore = (
            :BokhovenIterativeAlgorithm,
            :InteriorPointMethod,
            :LCP,
            :LCS,
            :LinearComplementarityAdjoint,
            :LinearComplementarityProblem,
            :LinearComplementaritySolution,
            :MCP,
            :MLCP,
            :MixedComplementarityAdjoint,
            :MixedComplementarityProblem,
            :MixedComplementaritySolution,
            :MixedLinearComplementarityProblem,
            :NCP,
            :NaiveLCSAlgorithm,
            :NonlinearComplementarityProblem,
            :NonlinearReformulation,
            :PATHSolverAlgorithm,
            :PGS,
            :PSOR,
            :RPGS,
            :RPSOR,
        ),
    ),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # PATHSolver C-API surface: no exports / no `public` declarations.
                :MCP_MajorIterationLimit, :MCP_MinorIterationLimit,
                :MCP_NoProgress, :MCP_Solved, :MCP_TimeLimit, :solve_mcp,
                :jacobian,                             # ForwardDiff documented-but-not-`public` API
            ),
        ),
        all_explicit_imports_are_public = (;
            # FillArrays.AbstractFill is documented but neither exported nor `public`.
            ignore = (:AbstractFill,),
        ),
    ),
)

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
    # Intentional, documented type piracy on ArrayInterfaceCore.can_setindex for
    # FillArrays.AbstractFill and Zygote.OneElement (src/ComplementaritySolve.jl).
    # Tracked at https://github.com/SciML/ComplementaritySolve.jl/issues/68
    aqua_broken = (:piracies,),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :Fix1, :Fix2,                          # Base (public in Julia 1.12+)
                :Infeasible, :MaxIters, :MaxTime,      # SciMLBase.ReturnCode
                :Success, :T, :Terminated,             # SciMLBase.ReturnCode
                :NullParameters,                       # SciMLBase
                :MCP_MajorIterationLimit, :MCP_MinorIterationLimit,  # PATHSolver
                :MCP_NoProgress, :MCP_Solved, :MCP_TimeLimit, :solve_mcp,  # PATHSolver
                :OneElement,                           # Zygote
                :can_setindex,                         # ArrayInterfaceCore
                :jacobian,                             # ForwardDiff
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@truncate_stacktrace"),        # TruncatedStacktraces
                :AbstractFill,                         # FillArrays
                :init, :solve, Symbol("solve!"),       # CommonSolve
            ),
        ),
    ),
)

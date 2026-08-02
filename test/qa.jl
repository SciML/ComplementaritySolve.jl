using SciMLTesting, ComplementaritySolve, Test

# PATHSolver declares no `export` and no `public` names at all, so every access into it
# trips `all_qualified_accesses_are_public`. Its README nevertheless directs users to
# `PATHSolver.solve_mcp` and the `MCP_Termination` enum values as the supported entry
# point, and there is no public alternative to reach the PATH C API. Drop this ignore
# once PATHSolver declares these names public upstream.
run_qa(
    ComplementaritySolve;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :MCP_MajorIterationLimit, :MCP_MinorIterationLimit,
                :MCP_NoProgress, :MCP_Solved, :MCP_TimeLimit, :solve_mcp,
            ),
        ),
    ),
)

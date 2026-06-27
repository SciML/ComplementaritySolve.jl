using Aqua, ComplementaritySolve

Aqua.test_all(
    ComplementaritySolve;
    ambiguities = false,    # Too many ambiguities from downstream
    persistent_tasks = false,  # PATHSolver precompile workload triggers persistent task detection
    project_extras = false, # Not sure about this one
    deps_compat = false
)    # Compat when we finally release it!

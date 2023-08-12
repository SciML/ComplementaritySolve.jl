using Aqua, ComplementaritySolve

Aqua.test_all(ComplementaritySolve;
    ambiguities=false,    # Too many ambiguities from downstream
    project_extras=false, # Not sure about this one
    deps_compat=false)    # Compat when we finally release it!

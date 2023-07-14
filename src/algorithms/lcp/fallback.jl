@views function solve(prob::LinearComplementarityProblem{iip, true},
    solver::AbstractComplementarityAlgorithm;
    kwargs...) where {iip}
    @warn "Solver: $(nameof(typeof(solver))) doesn't support batched problems. Falling \
           back to iterating over the batch dimension. This is going to be slow!!" maxlog=1
    # Do the first solve to get the types
    sol_first = solve(LinearComplementarityProblem{iip}(prob.M[:, :, 1],
            prob.q[:, 1],
            prob.u0[:, 1]),
        solver;
        kwargs...)

    zs = similar(sol_first.z, length(sol_first.z), size(prob.u0, 2))
    ws = similar(sol_first.w, length(sol_first.w), size(prob.u0, 2))
    # Residual might be nothing
    residual = sol_first.resid === nothing ? nothing :
               similar(sol_first.resid, length(sol_first.resid), size(prob.u0, 2))

    Threads.@threads for i in 1:size(prob.u0, 2)
        sol = solve(LinearComplementarityProblem{iip}(prob.M[:, :, i],
                prob.q[:, i],
                prob.u0[:, i]),
            solver;
            kwargs...)
        zs[:, i] .= sol.z
        ws[:, i] .= sol.w
        residual !== nothing && (residual[:, i] .= sol.resid)
    end

    return LinearComplementaritySolution(zs, ws, residual, prob, solver)
end

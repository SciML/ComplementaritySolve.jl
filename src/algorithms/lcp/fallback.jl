@views function __solve(
        prob::LinearComplementarityProblem{iip, true},
        solver::AbstractComplementarityAlgorithm, u0, M, q; verbose::Bool = true,
        kwargs...
    ) where {iip}
    if verbose
        @warn "Solver: $(nameof(typeof(solver))) doesn't support batched problems. Falling \
            back to iterating over the batch dimension. This is going to be slow!!" maxlog = 1
    end
    # Do the first solve to get the types
    ℒ𝒞𝒫 = LinearComplementarityProblem{iip, false}(nothing, nothing, nothing)
    sol_first = __solve(ℒ𝒞𝒫, solver, u0[:, 1], M[:, :, 1], q[:, 1]; kwargs...)

    us = similar(sol_first.u, length(sol_first.u), size(u0, 2))
    # Residual might be nothing
    residual = sol_first.residual === nothing ? nothing :
        similar(sol_first.residual, length(sol_first.residual), size(u0, 2))
    return_codes = Vector{ReturnCode.T}(undef, size(u0, 2))
    return_codes[1] = sol_first.retcode

    Threads.@threads :static for i in 1:size(u0, 2)
        sol = __solve(
            ℒ𝒞𝒫, solver, u0[:, mod1(i, size(u0, 2))],
            M[:, :, mod1(i, size(M, 3))], q[:, mod1(i, size(q, 2))]; kwargs...
        )
        us[:, i] .= sol.u
        residual !== nothing && (residual[:, i] .= sol.residual)
        return_codes[i] = sol.retcode
    end

    return LinearComplementaritySolution(us, residual, prob, solver, maximum(return_codes))
end
